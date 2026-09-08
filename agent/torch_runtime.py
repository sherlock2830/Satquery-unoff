"""
PyTorch runtime for the specialists trained on IndiaSat.

The Kaggle path exports ONNX; these CPU-trained models are served straight from
their `.pt` state dicts. Keeping this in its own module means `tools.py` stays a
thin dispatcher and torch is imported only when a torch-backed model is actually
called -- the API starts fast even though torch takes seconds to import.
"""
from __future__ import annotations

import os
import re
from typing import Any

WORLDCOVER_NAMES = ["tree cover", "shrubland", "grassland", "cropland",
                    "built-up", "bare or sparse vegetation", "snow and ice",
                    "permanent water bodies", "herbaceous wetland",
                    "mangroves", "moss and lichen"]
CHANGE_NAMES = ["increased", "decreased", "unchanged"]

_MODELS: dict[str, Any] = {}
_NORM: dict[str, Any] | None = None


def _norm() -> dict[str, Any]:
    """Training-split normalisation statistics.

    Inference must reuse exactly the numbers the models were fitted with.
    Recomputing them from whatever image a user happens to supply would shift
    the input distribution and quietly degrade every prediction -- a failure
    with no error message, which is the worst kind.
    """
    global _NORM
    if _NORM is None:
        import numpy as np
        from agent.registry import WEIGHTS_DIR
        p = os.path.join(WEIGHTS_DIR, "indiasat_norm.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(
                "models/weights/indiasat_norm.npz missing — "
                "run scripts/export_norm.py")
        z = np.load(p, allow_pickle=True)
        _NORM = {"m2": z["mean_s2"], "s2": z["std_s2"],
                 "m1": z["mean_s1"], "s1": z["std_s1"],
                 "vocab": [str(v) for v in z["vocab"]]}
    return _NORM


def stack_for(path: str):
    """Normalised (S2 12-band, S1 2-band) at 120x120, plus a `degraded` flag.

    AOI fetches write `<stem>.tif` (8-bit RGB, for display) beside `<stem>.npz`
    (the real band cube). The agent is handed the .tif, so the sidecar is how a
    model reaches the bands it was trained on. With no sidecar — a plain user
    upload — RGB is mapped into the blue/green/red slots and the remaining nine
    bands are held at the dataset mean. That is a degraded input, and the flag
    makes sure the answer says so rather than pretending otherwise.
    """
    import numpy as np
    from PIL import Image

    n = _norm()
    npz = path.rsplit(".", 1)[0] + ".npz"
    s2 = s1 = None
    if os.path.exists(npz):
        z = np.load(npz)
        s2 = z["s2"].astype(np.float32) if "s2" in z else None
        s1 = z["s1"].astype(np.float32) if "s1" in z else None

    def resize(a, c):
        out = np.zeros((c, 120, 120), np.float32)
        for i in range(min(c, a.shape[0])):
            out[i] = np.asarray(
                Image.fromarray(a[i]).resize((120, 120), Image.BILINEAR))
        return out

    degraded = s2 is None
    if degraded:
        im = Image.open(path).convert("RGB").resize((120, 120), Image.BILINEAR)
        rgb = np.asarray(im, np.float32).transpose(2, 0, 1)          # R, G, B
        s2 = np.tile(n["m2"][:, None, None], (1, 120, 120)).astype(np.float32)
        s2[1] = rgb[2] / 255.0 * 3000        # blue
        s2[2] = rgb[1] / 255.0 * 3000        # green
        s2[3] = rgb[0] / 255.0 * 3000        # red
    else:
        s2 = resize(s2, 12)

    s1 = (resize(s1, 2) if s1 is not None
          else np.tile(n["m1"][:, None, None], (1, 120, 120)).astype(np.float32))

    s2n = (s2 - n["m2"][:, None, None]) / n["s2"][:, None, None]
    s1n = (s1 - n["m1"][:, None, None]) / n["s1"][:, None, None]
    return s2n, s1n, degraded


def _model(spec):
    import torch
    if spec.id not in _MODELS:
        from models.backbone import (M1Clip, M3Caption, M4Ground, M5Change,
                                     M6Fusion)
        v = len(_norm()["vocab"])
        ctor = {"M1": lambda: M1Clip(v), "M3": lambda: M3Caption(v),
                "M4": lambda: M4Ground(v), "M5a": lambda: M5Change(v, 3),
                "M5b": lambda: M5Change(v, 3), "M6": M6Fusion}[spec.id]
        m = ctor()
        m.load_state_dict(torch.load(spec.weights_path, map_location="cpu"))
        m.eval()
        _MODELS[spec.id] = m
    return _MODELS[spec.id]


_M7: dict[str, Any] = {}


def _m7():
    """Load M7 and its own vocabulary.

    M7 carries a separate vocab file because it needs one token the shared
    IndiaSat vocabulary does not have -- <sep>, the instruction/answer
    boundary. Appending it to the shared file would change len(vocab) and
    break the embedding shape of every other model's checkpoint, so M7 keeps
    its own and the base 264 indices stay identical.
    """
    import json
    import torch
    if not _M7:
        from agent.registry import WEIGHTS_DIR, get
        from models.vlm import M7VLM
        with open(os.path.join(WEIGHTS_DIR, "m7_vlm_vocab.json"),
                  encoding="utf-8") as fh:
            meta = json.load(fh)
        m = M7VLM(vocab=len(meta["vocab"]))
        m.load_state_dict(torch.load(get("M7").weights_path, map_location="cpu"))
        m.eval()
        _M7.update({"model": m, "meta": meta,
                    "w2i": {w: i for i, w in enumerate(meta["vocab"])}})
    return _M7


def _m7_answer(query: str, path: str, max_tokens: int = 48) -> tuple[str, bool]:
    """Greedy generation from M7. Returns (text, degraded_input)."""
    import numpy as np
    import torch

    h = _m7()
    m, meta = h["model"], h["meta"]
    s2, s1, degraded = stack_for(path)
    x = torch.from_numpy(np.concatenate([s2, s1], 0))[None]
    toks = re.findall(r"[a-z0-9]+", query.lower())[:meta["max_instr"]]
    ids = [meta["bos"]] + [h["w2i"].get(t, 1) for t in toks] + [meta["sep"]]
    with torch.no_grad():
        pooled, grid = m.encode_image(x)
        out = m.generate(pooled, grid, torch.tensor([ids]), eos=meta["eos"],
                         max_new=max_tokens)
    vocab = meta["vocab"]
    txt = " ".join(vocab[i] for i in out[0].tolist()
                   if 3 < i < meta["sep"])
    return txt, degraded


def _encode(text: str, maxlen: int = 32):
    import torch
    vocab = _norm()["vocab"]
    w2i = {w: i for i, w in enumerate(vocab)}
    toks = re.findall(r"[a-z0-9]+", text.lower())[:maxlen]
    ids = [w2i.get(t, 1) for t in toks]
    mask = [1.0] * len(ids) + [0.0] * (maxlen - len(ids))
    ids += [0] * (maxlen - len(ids))
    return torch.tensor([ids[:maxlen]]), torch.tensor([mask[:maxlen]])


def run(spec, query: str, images: list[dict[str, Any]],
        params: dict[str, Any], **kw: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    if spec.id == "M7":
        return _run_m7(query, images, params, **kw)

    if not images:
        return {"stub": True, "summary": "[no image supplied]",
                "confidence": None, "evidence": []}

    model = _model(spec)
    s2, s1, degraded = stack_for(images[0]["path"])
    note = "  [RGB-only input: 9 of 12 bands unavailable]" if degraded else ""
    x14 = torch.from_numpy(np.concatenate([s2, s1], 0))[None]

    with torch.no_grad():
        if spec.id == "M6":
            p = torch.sigmoid(model(torch.from_numpy(s2)[None],
                                    torch.from_numpy(s1)[None]))[0].numpy()
            order = np.argsort(-p)
            hits = [f"{WORLDCOVER_NAMES[i]} {p[i]:.2f}"
                    for i in order[:4] if p[i] >= 0.5]
            if not hits:
                hits = [f"{WORLDCOVER_NAMES[order[0]]} {p[order[0]]:.2f} "
                        f"(below 0.5 threshold)"]
            return {"stub": False,
                    "summary": "land cover: " + ", ".join(hits) + note,
                    "confidence": float(p[order[0]]),
                    "classes": {WORLDCOVER_NAMES[i]: round(float(p[i]), 4)
                                for i in order},
                    "evidence": []}

        if spec.id == "M3":
            g = model.generate(x14, bos=2, eos=3, max_len=48)[0].tolist()
            vocab = _norm()["vocab"]
            txt = " ".join(vocab[i] for i in g if 3 < i < len(vocab))
            return {"stub": False, "answer": txt,
                    "summary": (txt[:220] or "(empty caption)") + note,
                    "confidence": None, "evidence": []}

        if spec.id == "M4":
            ids, mask = _encode(query)
            b = model(x14, ids, mask)[0].numpy()
            box = [round(float(v), 3) for v in b]
            return {"stub": False, "box": box,
                    "summary": f"box [{box[0]}, {box[1]}, {box[2]}, {box[3]}] "
                               f"(normalised x0,y0,x1,y1)" + note,
                    "confidence": None, "evidence": []}

        if spec.id in ("M5a", "M5b"):
            if len(images) < 2:
                return {"stub": True,
                        "summary": "[change analysis needs two images]",
                        "confidence": None, "evidence": []}
            a2, _, _ = stack_for(images[0]["path"])
            b2, _, _ = stack_for(images[1]["path"])
            ids, mask = _encode(query)
            lm, la = model(torch.from_numpy(a2)[None],
                           torch.from_numpy(b2)[None], ids, mask)
            changed = float((torch.sigmoid(lm) > 0.5).float().mean())
            if spec.id == "M5a":
                return {"stub": False, "changed_fraction": round(changed, 4),
                        "summary": f"{changed*100:.1f}% of the scene changed"
                                   + note,
                        "confidence": None, "evidence": []}
            p = torch.softmax(la, 1)[0].numpy()
            k = int(p.argmax())
            # M5b is supervised on dNDVI, so it measures VEGETATION change and
            # nothing else. Asked about built-up area it still answers about
            # vegetation, which reads as an answer to the question unless the
            # scope is stated. Say it whenever the question is not about
            # vegetation, so the trace cannot mislead.
            asked_veg = bool(re.search(r"vegetat|green|crop|forest|ndvi",
                                       query, re.I))
            scope = ("" if asked_veg else
                     "  [scope: this model is trained on NDVI difference and "
                     "reports VEGETATION change only — it was not trained to "
                     "answer about built-up area or other classes]")
            return {"stub": False, "answer": CHANGE_NAMES[k],
                    "measures": "vegetation (dNDVI)",
                    "summary": f"vegetation {CHANGE_NAMES[k]} "
                               f"({changed*100:.1f}% of pixels changed)"
                               + note + scope,
                    "confidence": float(p[k]), "evidence": []}

        if spec.id == "M1":
            emb = model.encode_image(x14)[0].numpy()
            return {"stub": False, "summary": f"embedding dim {emb.shape[0]}",
                    "confidence": None, "evidence": []}

    return {"stub": True, "summary": f"[no torch adapter for {spec.id}]",
            "confidence": None, "evidence": []}


# --------------------------------------------------------------------------- #
# M7 -- narration, and only narration
# --------------------------------------------------------------------------- #
# Words that only ever appear inside a "<class> covers approximately N ..."
# clause in an IndiaSat caption. Stripping backwards through them is what
# leaves a clean sentence after the generated figure is removed.
_FIGURE_FILLER = {
    "covers", "cover", "covering", "approximately", "about", "around",
    "of", "the", "scene", "image", "alongside", "and", "with", "plus",
    "built", "up", "tree", "shrubland", "grassland", "cropland", "bare",
    "sparse", "vegetation", "permanent", "water", "bodies", "herbaceous",
    "wetland", "mangroves", "moss", "lichen", "snow", "ice",
}


def _strip_figures(text: str) -> str:
    """Remove every generated number from an M7 caption.

    IndiaSat captions are written as "<class> covers approximately N% of the
    scene", so M7 learned to emit percentages -- and it emits them from a
    120x120 thumbnail of a patch, not from the AOI the user actually asked
    about. Left in, they sit in the same paragraph as the measured figures and
    disagree with them, and a reader has no way to tell which is which.

    Truncating at the first digit and walking back through the clause that
    introduced it leaves the qualitative description, which is what M7 is
    actually for.
    """
    toks = text.split()
    cut = next((i for i, t in enumerate(toks) if any(c.isdigit() for c in t)),
               len(toks))
    while cut > 0 and toks[cut - 1] in _FIGURE_FILLER:
        cut -= 1
    return " ".join(toks[:cut]).strip()


def _run_m7(query: str, images: list[dict[str, Any]], params: dict[str, Any],
            findings: list[str] | None = None,
            measured: list[str] | None = None,
            dominant: str | None = None,
            **_: Any) -> dict[str, Any]:
    """Compose the final answer.

    Three ingredients, kept separate on purpose:

      1. M7's own sentence, generated from the image and the question. This is
         the only generated text in the answer.
      2. What each specialist actually returned, verbatim.
      3. The measured statistics, copied from serve/analysis.py.

    M7 never produces a number that appears in the report. A 2.34 M-parameter
    decoder trained on one corpus is good enough to describe a scene and bad
    enough that a percentage it invented would be indistinguishable from one
    that was measured -- so it is not allowed to supply one.
    """
    parts: list[str] = []
    generated = ""
    if images:
        try:
            max_tok = int(params.get("max_tokens", 48))
            generated, degraded = _m7_answer(query, images[0]["path"], max_tok)

            # M7 is trained on IndiaSat instructions, most of which are yes/no
            # or multiple choice, so a long compound question can pull a bare
            # "yes" out of it. A one-word reply to a question that was not a
            # yes/no question is not an answer -- fall back to the scene
            # description instruction, which is in its training distribution,
            # and say that is what the sentence is.
            polar = bool(re.match(r"\s*(is|are|do|does|would|can|has|have)\b",
                                  query, re.I))
            if len(generated.split()) <= 2 and not polar:
                described, degraded = _m7_answer(
                    "Provide a detailed scene description for this remote "
                    "sensing image.", images[0]["path"], max_tok)
                # If the fallback is degenerate too, say nothing rather than
                # label two words as a scene description. The measured table
                # below carries the answer either way.
                if len(described.split()) > 3:
                    generated = described

            text = _strip_figures(generated)
            if text and len(text.split()) > 3:
                # Labelled as generated, every time. The place name and the
                # dominant class are M7's predictions from a 120x120 patch --
                # plausible, frequently right, and not evidence.
                parts.append("Scene description (M7, generated — the location "
                             "and land-cover class are model predictions, not "
                             "measurements): " + text + ".")
            elif text:
                parts.append(text[0].upper() + text[1:])

            if parts and degraded:
                parts[-1] += "  [RGB-only input: 9 of 12 bands unavailable]"

            # VERIFY, in the sense PS 26167 asks for: where the generated
            # description and the measured pixels disagree about what the
            # scene mostly is, say so rather than printing both and leaving
            # the reader to notice.
            if dominant and text:
                claimed = re.search(
                    r"predominantly ([a-z ]+?) landscape", text)
                if claimed and dominant not in claimed.group(1).strip():
                    parts.append(
                        f"⚠ Disagreement: M7 describes the scene as "
                        f"predominantly {claimed.group(1).strip()}, while the "
                        f"measured indices make {dominant} the largest class. "
                        f"The measured value is the one to act on.")
        except Exception as exc:
            parts.append(f"[M7 unavailable: {type(exc).__name__}: {exc}]")

    if measured:
        parts.append("Measured from the pixels: " + " ".join(measured))
    if findings:
        parts.append("Specialist findings: " + " · ".join(findings))
    if not parts:
        parts.append("[no image and no specialist output]")

    return {"stub": False, "answer": "\n\n".join(parts),
            "generated": generated, "summary": "synthesised",
            "confidence": None, "evidence": []}
