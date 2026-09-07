"""
Fetch live Sentinel imagery for a user-drawn area of interest.

The map console draws a rectangle; this turns it into real, co-registered
imagery the agent can analyse:

    bbox  ->  STAC search (Earth Search, keyless)
          ->  Sentinel-2 L2A window   (12 bands, optical)
          ->  Sentinel-1 GRD window   (VV/VH, SAR)      [optional]
          ->  Sentinel-2 at an earlier date             [optional, bi-temporal]

Each fetch writes two files per image, sharing a stem:

    <stem>.tif   8-bit RGB GeoTIFF -- georeferenced, PIL-readable, and what the
                 UI displays. This is the `path` handed to the agent.
    <stem>.npz   the full 12-band S2 (+2-band S1) float stack, for the models
                 that consume all bands.

Two files rather than one because the trained specialists need 12 bands, while
M2's ONNX graph and every image viewer expect 3. Keeping the stem shared means
a tool can find the full stack from the path it was given, without a registry.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AOI_DIR = os.path.join(ROOT, "data", "aoi_fetch")
STAC = "https://earth-search.aws.element84.com/v1/search"

# The same 12 bands, in the same order, that the trained models expect. A band
# missing from a scene becomes a zero channel rather than shifting every later
# index — silently reordering would be far worse than one dead channel.
S2_BANDS = ["coastal", "blue", "green", "red", "rededge1", "rededge2",
            "rededge3", "nir", "nir08", "nir09", "swir16", "swir22"]
S1_BANDS = ["vv", "vh"]
MAX_PX = 1024          # cap the fetch so one careless drag cannot pull a whole scene

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("VSI_CACHE", "TRUE")


def _stac(collection: str, geom: dict, start: str, end: str,
          extra: dict | None = None, limit: int = 20) -> list[dict]:
    body: dict[str, Any] = {"collections": [collection], "intersects": geom,
                            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
                            "limit": limit}
    if extra:
        body["query"] = extra
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["features"]


def _bbox_geom(bbox: list[float]) -> dict:
    w, s, e, n = bbox
    return {"type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def _window_for(href: str, bbox: list[float]):
    """Pixel window covering `bbox` inside the raster at `href`.

    Returns None when the AOI is not fully inside this scene -- a partially
    covered AOI yields a patch with a nodata wedge, which reads to a model as a
    large black region and produces confidently wrong answers.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window

    w, s, e, n = bbox
    with rasterio.open(href) as ds:
        xs, ys = warp_transform("EPSG:4326", ds.crs, [w, e], [s, n])
        r1, c1 = ds.index(xs[0], ys[1])       # top-left  (west, north)
        r2, c2 = ds.index(xs[1], ys[0])       # bottom-right
        row, col = min(r1, r2), min(c1, c2)
        h, wd = abs(r2 - r1), abs(c2 - c1)
        if h < 32 or wd < 32:
            return None, "AOI is too small — draw a larger box (min ~320 m)"
        if h > MAX_PX or wd > MAX_PX:
            return None, (f"AOI is too large ({wd}x{h} px at 10 m). "
                          f"Draw a box under ~{MAX_PX//100} km a side.")
        if row < 0 or col < 0 or row + h > ds.height or col + wd > ds.width:
            return None, None                 # not fully inside: try next scene
        return (Window(col, row, wd, h), ds.crs, ds.window_transform(
            Window(col, row, wd, h)), ds.width), None


def _pick(feats: list[dict], bbox: list[float], sort_key):
    """First scene that fully contains the AOI."""
    err = None
    for f in sorted(feats, key=sort_key):
        href = f["assets"].get("red", {}).get("href") or \
               f["assets"].get("vv", {}).get("href")
        if not href:
            continue
        try:
            got, e = _window_for(href, bbox)
        except Exception:
            continue
        if e:
            err = e                            # size complaint: report it
            break
        if got:
            return f, got, None
    return None, None, err


def _read_bands(scene: dict, bands: list[str], win, refw: int,
                out_hw: tuple[int, int], dtype, workers: int = 8) -> np.ndarray:
    """Read every band into one cube, fetching bands concurrently.

    Each band is a separate COG on S3, so the reads are independent and almost
    entirely network-bound. Serially this took ~240 s for a 4 km AOI over a
    2.5 MB/s link -- far too slow for someone drawing a box and waiting. Each
    thread opens its own dataset handle, which is the supported way to use
    rasterio concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor

    import rasterio
    from rasterio.warp import Resampling
    from rasterio.windows import Window

    H, W = out_hw
    cube = np.zeros((len(bands), H, W), dtype=dtype)

    def grab(i_b):
        i, b = i_b
        href = scene["assets"].get(b, {}).get("href")
        if not href:
            return i, None
        with rasterio.open(href) as ds:
            sc = ds.width / refw
            sw = Window(win.col_off * sc, win.row_off * sc,
                        win.width * sc, win.height * sc)
            return i, ds.read(1, window=sw, out_shape=(H, W),
                              resampling=Resampling.bilinear,
                              boundless=True, fill_value=0)

    with ThreadPoolExecutor(max_workers=min(workers, len(bands))) as ex:
        for i, arr in ex.map(grab, list(enumerate(bands))):
            if arr is not None:
                cube[i] = arr
    return cube


def _stretch(band: np.ndarray, lo_pct=2.0, hi_pct=98.0) -> np.ndarray:
    """Percentile stretch to 8-bit for display.

    Sentinel-2 reflectance occupies a narrow slice of uint16, so a plain
    linear scale renders almost black. Percentile clipping is what makes the
    preview look like the satellite image a user expects.
    """
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1
    return np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _write_pair(stem: str, rgb: np.ndarray, crs, transform,
                stack: dict[str, np.ndarray]) -> str:
    """Write <stem>.tif (8-bit RGB, georeferenced) and <stem>.npz (full stack)."""
    import rasterio

    path = f"{stem}.tif"
    with rasterio.open(path, "w", driver="GTiff", height=rgb.shape[1],
                       width=rgb.shape[2], count=3, dtype="uint8",
                       crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(rgb)
    np.savez_compressed(f"{stem}.npz", **stack)
    return path


def fetch_aoi(bbox: list[float], start: str, end: str, max_cloud: int = 20,
              want_sar: bool = True, start2: str | None = None,
              end2: str | None = None,
              sar_budget_s: int = 45) -> dict[str, Any]:
    """Fetch imagery for one AOI. Returns image refs + scene provenance."""
    import rasterio

    os.makedirs(AOI_DIR, exist_ok=True)
    geom = _bbox_geom(bbox)
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid() % 9973:04d}"
    out: dict[str, Any] = {"images": [], "scenes": [], "aoi": bbox}

    feats = _stac("sentinel-2-l2a", geom, start, end,
                  {"eo:cloud_cover": {"lt": max_cloud}})
    if not feats:
        return {"error": f"No Sentinel-2 scene under {max_cloud}% cloud between "
                         f"{start} and {end}. Widen the dates or raise the "
                         f"cloud limit."}
    s2, got, err = _pick(feats, bbox,
                         lambda f: f["properties"]["eo:cloud_cover"])
    if err:
        return {"error": err}
    if s2 is None:
        return {"error": (f"{len(feats)} scenes overlap this area but none "
                          f"covers it completely. Draw a smaller box, or move "
                          f"it away from a scene edge.")}

    win, crs, transform, refw = got
    H, W = int(win.height), int(win.width)

    cube = _read_bands(s2, S2_BANDS, win, refw, (H, W), np.uint16)
    rgb = np.stack([_stretch(cube[3]), _stretch(cube[2]), _stretch(cube[1])])
    stem = os.path.join(AOI_DIR, f"{stamp}_s2")
    path = _write_pair(stem, rgb, crs, transform, {"s2": cube})
    date = s2["properties"]["datetime"][:10]
    out["images"].append({"path": path, "modality": "optical", "date": date,
                          "role": "optical (t1)"})
    out["scenes"].append({"id": s2["id"], "collection": "sentinel-2-l2a",
                          "date": date,
                          "cloud": round(s2["properties"]["eo:cloud_cover"], 2)})

    # --- Sentinel-1 SAR over the same footprint --------------------------- #
    # Bounded by a wall-clock budget. A Sentinel-1 GRD product is a multi-GB
    # COG and, on a slow link, opening one can take minutes -- measured at
    # >260 s here for a single scene. SAR is valuable but optional, so a slow
    # fetch degrades to optical-only with a stated reason rather than hanging
    # the request.
    if want_sar:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

        def _grab_sar():
            sf = _stac("sentinel-1-grd", geom, start, end)
            if not sf:
                return None, None
            s1, got1, _ = _pick(sf, bbox, lambda f: f["properties"]["datetime"])
            return s1, got1

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_grab_sar)
            try:
                s1, got1 = fut.result(timeout=sar_budget_s)
            except FTimeout:
                s1 = got1 = None
                out["warnings"] = out.get("warnings", []) + [
                    f"Sentinel-1 lookup exceeded {sar_budget_s}s and was skipped "
                    f"(GRD products are multi-GB). Optical-only results below."]
            except Exception as exc:
                s1 = got1 = None
                out["warnings"] = out.get("warnings", []) + [
                    f"Sentinel-1 lookup failed: {type(exc).__name__}"]
        if s1 and got1:
            w1, crs1, tr1, refw1 = got1
            sar = _read_bands(s1, S1_BANDS, w1, refw1,
                              (int(w1.height), int(w1.width)), np.float32)
            # dB scaling: SAR backscatter is heavy-tailed, so a linear stretch
            # is nearly black. dB is also what analysts read.
            db = 10 * np.log10(np.clip(sar, 1e-3, None))
            vis = np.stack([_stretch(db[0]), _stretch(db[1]),
                            _stretch(db[0])])
            stem1 = os.path.join(AOI_DIR, f"{stamp}_s1")
            p1 = _write_pair(stem1, vis, crs1, tr1, {"s1": sar})
            d1 = s1["properties"]["datetime"][:10]
            out["images"].append({"path": p1, "modality": "sar", "date": d1,
                                  "role": "SAR"})
            out["scenes"].append({"id": s1["id"], "collection": "sentinel-1-grd",
                                  "date": d1, "cloud": None})
        else:
            out["warnings"] = out.get("warnings", []) + \
                ["No Sentinel-1 scene fully covers this AOI in the date range."]

    # --- earlier Sentinel-2 date for change analysis ---------------------- #
    if start2 and end2:
        f2 = _stac("sentinel-2-l2a", geom, start2, end2,
                   {"eo:cloud_cover": {"lt": max_cloud}})
        s2b, got2, _ = _pick(f2, bbox,
                             lambda f: f["properties"]["eo:cloud_cover"])
        if s2b and got2 and s2b["id"] != s2["id"]:
            w2, crs2, tr2, refw2 = got2
            c2 = _read_bands(s2b, S2_BANDS, w2, refw2,
                             (int(w2.height), int(w2.width)), np.uint16)
            rgb2 = np.stack([_stretch(c2[3]), _stretch(c2[2]), _stretch(c2[1])])
            stem2 = os.path.join(AOI_DIR, f"{stamp}_s2t2")
            p2 = _write_pair(stem2, rgb2, crs2, tr2, {"s2": c2})
            d2 = s2b["properties"]["datetime"][:10]
            # earlier date first, so the pair reads t1 -> t2 chronologically
            out["images"].insert(0, {"path": p2, "modality": "optical",
                                     "date": d2, "role": "optical (earlier)"})
            out["scenes"].append({"id": s2b["id"],
                                  "collection": "sentinel-2-l2a", "date": d2,
                                  "cloud": round(s2b["properties"]["eo:cloud_cover"], 2)})
        else:
            out["warnings"] = out.get("warnings", []) + \
                ["No distinct earlier Sentinel-2 scene found for this AOI."]

    out["size_px"] = [W, H]
    out["ground_km"] = [round(W * 10 / 1000, 2), round(H * 10 / 1000, 2)]
    return out
