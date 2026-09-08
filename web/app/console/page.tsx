"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Nav } from "@/components/Chrome";
import { AoiMap } from "@/components/console/AoiMap";
import { Results } from "@/components/console/Results";
import {
  api,
  EXAMPLE_QUERIES,
  type Health,
  type ImageRef,
  type Model,
  type RunSummary,
  type Scene,
  type Trace,
} from "@/lib/api";

type Backend =
  | { kind: "checking" }
  | { kind: "down"; error: string }
  | { kind: "up"; health: Health; models: Model[] };

const today = () => new Date().toISOString().slice(0, 10);
const yearAgo = (n: number) => {
  const d = new Date();
  d.setFullYear(d.getFullYear() - n);
  return d.toISOString().slice(0, 10);
};

export default function ConsolePage() {
  const [backend, setBackend] = useState<Backend>({ kind: "checking" });
  const [runs, setRuns] = useState<RunSummary[]>([]);

  // inputs
  const [source, setSource] = useState<"aoi" | "upload">("aoi");
  const [bbox, setBbox] = useState<number[] | null>(null);
  const [start, setStart] = useState(yearAgo(1));
  const [end, setEnd] = useState(today());
  const [bitemporal, setBitemporal] = useState(true);
  const [start2, setStart2] = useState(yearAgo(3));
  const [end2, setEnd2] = useState(yearAgo(2));
  const [cloud, setCloud] = useState(20);
  const [wantSar, setWantSar] = useState(false);

  // state
  const [images, setImages] = useState<ImageRef[]>([]);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [query, setQuery] = useState<string>(EXAMPLE_QUERIES[1].text);
  const [busy, setBusy] = useState<null | "fetch" | "upload" | "query">(null);
  const [error, setError] = useState<string | null>(null);
  const [trace, setTrace] = useState<Trace | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const loadBackend = useCallback(async () => {
    setBackend({ kind: "checking" });
    try {
      const [health, models] = await Promise.all([api.health(), api.models()]);
      setBackend({ kind: "up", health, models });
      api.runs(8).then(setRuns).catch(() => {});
    } catch (e) {
      setBackend({ kind: "down", error: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  useEffect(() => {
    void loadBackend();
  }, [loadBackend]);

  /* ------------------------------------------------------------ actions */
  const fetchAoi = async () => {
    if (!bbox) {
      setError("Draw an area of interest on the map first (shift + drag), or pick a preset.");
      return;
    }
    setBusy("fetch");
    setError(null);
    setWarnings([]);
    try {
      const r = await api.aoiFetch({
        bbox,
        start,
        end,
        max_cloud: cloud,
        want_sar: wantSar,
        start2: bitemporal ? start2 : null,
        end2: bitemporal ? end2 : null,
      });
      setImages(r.images);
      setScenes(r.scenes);
      setWarnings(r.warnings ?? []);
      setTrace(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    setBusy("upload");
    setError(null);
    try {
      const out: ImageRef[] = [];
      for (const f of Array.from(files).slice(0, 2)) {
        const r = await api.upload(f);
        out.push({
          path: r.path,
          modality: (r.modality as "optical" | "sar") ?? "optical",
          date: null,
          role: r.filename,
          preview: api.previewUrl(r.path),
        });
      }
      setImages(out);
      setScenes([]);
      setWarnings([
        "Uploaded images carry no band sidecar, so NDVI, NDBI and MNDWI cannot be computed — they need NIR and SWIR, which RGB does not have. Fetch an AOI from the map for measured percentages.",
      ]);
      setTrace(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const run = async () => {
    if (!query.trim()) return setError("Type a question first.");
    if (!images.length) return setError("Fetch or upload at least one image first.");
    setBusy("query");
    setError(null);
    try {
      const t = await api.query(query, images);
      setTrace(t);
      api.runs(8).then(setRuns).catch(() => {});
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const setModality = (i: number, m: "optical" | "sar") =>
    setImages((prev) => prev.map((im, j) => (j === i ? { ...im, modality: m } : im)));

  /* -------------------------------------------------------------- render */
  return (
    <>
      <Nav />
      <main id="main" className="mx-auto max-w-[1240px] px-5 pb-24 pt-[104px] sm:px-8">
        <div className="flex flex-wrap items-end justify-between gap-5">
          <div>
            <p className="eyebrow">Console</p>
            <h1 className="display mt-4 text-[clamp(1.7rem,3.6vw,2.5rem)] text-white">
              Draw an area. Ask a question.
            </h1>
            <p className="mt-4 max-w-[62ch] text-[14.5px] leading-[1.7] text-[var(--color-mute)]">
              Live Sentinel-1 and Sentinel-2 imagery, fetched for the box you draw, routed through the
              agent, and returned with measured land cover and the full execution trace.
            </p>
          </div>
          <BackendPill state={backend} onRetry={() => void loadBackend()} />
        </div>

        {backend.kind === "down" && <Offline error={backend.error} />}

        <div className="mt-10 grid gap-6 lg:grid-cols-[400px_1fr] lg:items-start">
          {/* ------------------------------------------------ input panel */}
          <div className="space-y-5 lg:sticky lg:top-[76px]">
            <div className="panel rounded-lg p-5">
              <div className="mb-4 flex gap-1 rounded-md border border-[var(--color-line)] p-1">
                {(["aoi", "upload"] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setSource(s)}
                    className={`mono flex-1 rounded px-3 py-1.5 text-[10.5px] tracking-[0.12em] transition-colors ${
                      source === s ? "bg-white/[0.09] text-white" : "text-[var(--color-mute)] hover:text-white"
                    }`}
                  >
                    {s === "aoi" ? "MAP AOI" : "UPLOAD"}
                  </button>
                ))}
              </div>

              {source === "aoi" ? (
                <>
                  <AoiMap bbox={bbox} onChange={setBbox} />

                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <DateField label="LATER FROM" value={start} onChange={setStart} />
                    <DateField label="LATER TO" value={end} onChange={setEnd} />
                  </div>

                  <label className="mt-4 flex cursor-pointer items-center gap-2.5">
                    <input
                      type="checkbox"
                      checked={bitemporal}
                      onChange={(e) => setBitemporal(e.target.checked)}
                      className="h-3.5 w-3.5 accent-[var(--color-brand)]"
                    />
                    <span className="text-[13px] text-[var(--color-dim)]">
                      Fetch an earlier date too (bi-temporal)
                    </span>
                  </label>

                  {bitemporal && (
                    <div className="mt-3 grid grid-cols-2 gap-3">
                      <DateField label="EARLIER FROM" value={start2} onChange={setStart2} />
                      <DateField label="EARLIER TO" value={end2} onChange={setEnd2} />
                    </div>
                  )}

                  <div className="mt-4">
                    <div className="flex items-center justify-between">
                      <span className="mono text-[10px] tracking-[0.14em] text-[var(--color-mute)]">
                        MAX CLOUD
                      </span>
                      <span className="mono text-[11px] tabular-nums text-white">{cloud}%</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={80}
                      value={cloud}
                      onChange={(e) => setCloud(Number(e.target.value))}
                      className="mt-2 w-full accent-[var(--color-brand)]"
                    />
                  </div>

                  <label className="mt-3 flex cursor-pointer items-start gap-2.5">
                    <input
                      type="checkbox"
                      checked={wantSar}
                      onChange={(e) => setWantSar(e.target.checked)}
                      className="mt-[3px] h-3.5 w-3.5 accent-[var(--color-brand)]"
                    />
                    <span className="text-[13px] leading-[1.5] text-[var(--color-dim)]">
                      Also fetch Sentinel-1 SAR
                      <span className="mono ml-1 block text-[10.5px] text-[var(--color-mute)]">
                        GRD products are multi-GB; the fetch is budgeted at 45 s and degrades to
                        optical-only rather than hanging.
                      </span>
                    </span>
                  </label>

                  <button
                    type="button"
                    onClick={() => void fetchAoi()}
                    disabled={busy !== null || backend.kind !== "up"}
                    className="mt-5 w-full rounded-md bg-[var(--color-brand)] px-4 py-2.5 text-[13.5px] font-medium text-white transition-colors hover:bg-[#4b8ef5] disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {busy === "fetch" ? "Fetching imagery…" : "Fetch imagery"}
                  </button>
                  {busy === "fetch" && (
                    <p className="mono mt-2 text-center text-[10.5px] text-[var(--color-mute)]">
                      reading 12 bands off S3 — 20–60 s
                    </p>
                  )}
                </>
              ) : (
                <>
                  <input
                    ref={fileInput}
                    type="file"
                    accept=".tif,.tiff,.png,.jpg,.jpeg"
                    multiple
                    hidden
                    onChange={(e) => void upload(e.target.files)}
                  />
                  <button
                    type="button"
                    onClick={() => fileInput.current?.click()}
                    disabled={busy !== null || backend.kind !== "up"}
                    className="w-full rounded-lg border border-dashed border-[var(--color-line)] px-4 py-10 text-center transition-colors hover:border-white/25 disabled:opacity-40"
                  >
                    <span className="block text-[14px] text-white">
                      {busy === "upload" ? "Uploading…" : "Choose one or two images"}
                    </span>
                    <span className="mono mt-2 block text-[10.5px] text-[var(--color-mute)]">
                      GEOTIFF · TIFF · PNG · JPEG · MAX 256 MB
                    </span>
                  </button>
                  <p className="mt-4 text-[12.5px] leading-[1.7] text-[var(--color-mute)]">
                    Two images means a pair: two dates for change analysis, or one optical and one SAR for
                    cross-modal extraction. Set the modality below after uploading.
                  </p>
                </>
              )}
            </div>

            {/* selected imagery */}
            {images.length > 0 && (
              <div className="panel rounded-lg p-5">
                <p className="mono text-[10px] tracking-[0.14em] text-[var(--color-mute)]">
                  SELECTED IMAGERY — {images.length}
                </p>
                <div className="mt-3 space-y-3">
                  {images.map((im, i) => (
                    <div key={im.path} className="flex gap-3">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={im.preview ?? api.previewUrl(im.path)}
                        alt={im.role ?? "scene"}
                        className="h-16 w-16 shrink-0 rounded border border-[var(--color-line)] object-cover"
                      />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-[12.5px] text-white">{im.role ?? `image ${i + 1}`}</p>
                        <p className="mono mt-0.5 text-[10.5px] text-[var(--color-mute)]">
                          {im.date ?? "no date"}
                        </p>
                        <div className="mt-1.5 flex gap-1">
                          {(["optical", "sar"] as const).map((m) => (
                            <button
                              key={m}
                              type="button"
                              onClick={() => setModality(i, m)}
                              className={`mono rounded px-2 py-[3px] text-[9.5px] tracking-[0.1em] transition-colors ${
                                im.modality === m
                                  ? "bg-[var(--color-brand)]/20 text-[var(--color-brand-soft)]"
                                  : "text-[var(--color-mute)] hover:text-white"
                              }`}
                            >
                              {m.toUpperCase()}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                {scenes.length > 0 && (
                  <div className="mt-4 border-t border-[var(--color-line)] pt-3">
                    <p className="mono text-[10px] tracking-[0.14em] text-[var(--color-mute)]">
                      SCENE PROVENANCE
                    </p>
                    {scenes.map((s) => (
                      <p key={s.id} className="mono mt-1.5 break-all text-[10px] leading-[1.6] text-[var(--color-mute)]">
                        {s.id}
                        <br />
                        {s.collection} · {s.date}
                        {s.cloud !== null && ` · ${s.cloud}% cloud`}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ----------------------------------------------- query + results */}
          <div className="space-y-5">
            <div className="panel rounded-lg p-5">
              <p className="mono text-[10px] tracking-[0.14em] text-[var(--color-mute)]">QUESTION</p>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={3}
                placeholder="Ask about the imagery in plain language…"
                className="mt-3 w-full resize-y rounded-md border border-[var(--color-line)] bg-white/[0.03] px-4 py-3 text-[14.5px] leading-[1.6] text-white outline-none placeholder:text-white/25 focus:border-[var(--color-brand)]"
              />
              <div className="mt-3 flex flex-wrap gap-1.5">
                {EXAMPLE_QUERIES.map((q) => (
                  <button
                    key={q.label}
                    type="button"
                    onClick={() => setQuery(q.text)}
                    className="mono rounded-full border border-[var(--color-line)] px-2.5 py-1 text-[10.5px] tracking-[0.08em] text-[var(--color-mute)] transition-colors hover:border-white/25 hover:text-white"
                  >
                    {q.label.toUpperCase()}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => void run()}
                disabled={busy !== null || backend.kind !== "up" || images.length === 0}
                className="mt-4 w-full rounded-md bg-white px-4 py-2.5 text-[13.5px] font-medium text-black transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-30"
              >
                {busy === "query" ? "Running the agent…" : "Run query"}
              </button>
            </div>

            {error && (
              <div className="rounded-lg border border-[var(--color-signal)]/35 bg-[var(--color-signal)]/[0.06] px-5 py-4">
                <p className="mono text-[10px] tracking-[0.14em] text-[var(--color-signal)]">ERROR</p>
                <p className="mt-2 text-[13.5px] leading-[1.65] text-white">{error}</p>
              </div>
            )}

            {warnings.map((w) => (
              <div key={w} className="rounded-lg border border-[var(--color-line)] px-5 py-4">
                <p className="mono text-[10px] tracking-[0.14em] text-[var(--color-mute)]">NOTE</p>
                <p className="mt-2 text-[13px] leading-[1.7] text-[var(--color-mute)]">{w}</p>
              </div>
            ))}

            <div ref={resultsRef}>
              {trace ? (
                <Results trace={trace} />
              ) : (
                <Placeholder backend={backend} hasImages={images.length > 0} />
              )}
            </div>

            {runs.length > 0 && (
              <div className="overflow-hidden rounded-lg border border-[var(--color-line)]">
                <p className="mono border-b border-[var(--color-line)] px-5 py-3 text-[10.5px] tracking-[0.14em] text-[var(--color-mute)]">
                  RECENT RUNS
                </p>
                {runs.map((r) => (
                  <div
                    key={r.run_id}
                    className="flex flex-wrap items-baseline justify-between gap-2 border-b border-[var(--color-line)] px-5 py-3 last:border-0"
                  >
                    <span className="min-w-0 flex-1 truncate text-[13px] text-[var(--color-dim)]">
                      {r.query}
                    </span>
                    <span className="mono text-[10.5px] text-[var(--color-mute)]">
                      {r.rejected ? "refused" : (r.models_invoked ?? []).join(",")} · {r.total_ms} ms
                    </span>
                    <a
                      href={api.reportUrl(r.run_id)}
                      download
                      className="mono text-[10.5px] text-[var(--color-brand-soft)] hover:underline"
                    >
                      .md
                    </a>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </>
  );
}

/* ------------------------------------------------------------ fragments */
function DateField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="mono text-[10px] tracking-[0.14em] text-[var(--color-mute)]">{label}</span>
      <input
        type="date"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mono mt-1.5 w-full rounded-md border border-[var(--color-line)] bg-white/[0.03] px-3 py-2 text-[12px] text-white outline-none [color-scheme:dark] focus:border-[var(--color-brand)]"
      />
    </label>
  );
}

function BackendPill({ state, onRetry }: { state: Backend; onRetry: () => void }) {
  const [dot, text] =
    state.kind === "up"
      ? ["bg-[var(--color-verd)]", `${state.health.models_trained.length}/${state.health.models_registered} MODELS LOADED`]
      : state.kind === "checking"
        ? ["bg-[var(--color-mute)]", "CHECKING BACKEND"]
        : ["bg-[var(--color-signal)]", "BACKEND OFFLINE"];
  return (
    <button
      type="button"
      onClick={onRetry}
      className="mono inline-flex items-center gap-2.5 rounded-full border border-[var(--color-line)] px-4 py-[7px] text-[11px] tracking-[0.12em] text-[var(--color-dim)] transition-colors hover:border-white/25"
    >
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {text}
    </button>
  );
}

function Offline({ error }: { error: string }) {
  return (
    <div className="mt-8 rounded-lg border border-[var(--color-signal)]/35 bg-[var(--color-signal)]/[0.06] p-6">
      <p className="text-[15px] text-white">The inference service is not reachable.</p>
      <p className="mono mt-2 text-[11.5px] text-[var(--color-signal)]">{error}</p>
      <p className="mt-4 text-[13.5px] leading-[1.7] text-[var(--color-mute)]">
        The models run on CPU behind a FastAPI service; they are not deployed to this host, because the
        weights and the Sentinel fetch need a long-lived process rather than a serverless function. Clone
        the repository and start it:
      </p>
      <pre className="mono mt-3 overflow-x-auto rounded bg-white/[0.04] px-4 py-3 text-[11.5px] leading-[1.8] text-[var(--color-dim)]">
        {`git clone https://github.com/sherlock2830/Satquery-unoff
cd Satquery-unoff
pip install -r requirements.txt
python -m models.train_all && python -m models.train_vlm
uvicorn serve.api:app --port 8000`}
      </pre>
      <p className="mt-3 text-[12.5px] leading-[1.7] text-[var(--color-mute)]">
        Then set <span className="mono">SATQUERY_API</span> to its URL, or run the site locally with{" "}
        <span className="mono">npm --prefix web run dev</span> — the rewrite in next.config.ts points at{" "}
        <span className="mono">127.0.0.1:8000</span> by default.
      </p>
    </div>
  );
}

function Placeholder({ backend, hasImages }: { backend: Backend; hasImages: boolean }) {
  const steps = [
    ["INGEST", "modality, bands, projection, GSD, dates"],
    ["VALIDATE", "can these inputs answer this question?"],
    ["ROUTE", "task family → specialists from the registry"],
    ["EXECUTE", "run the selected models in order"],
    ["MEASURE", "NDVI / MNDWI / NDBI over the real pixels"],
    ["SYNTHESISE", "M7 narrates; it never supplies a number"],
    ["EXPLAIN", "answer + measured table + execution trace"],
  ];
  return (
    <div className="rounded-lg border border-dashed border-[var(--color-line)] p-8">
      <p className="mono text-[10.5px] tracking-[0.14em] text-[var(--color-mute)]">
        {backend.kind !== "up"
          ? "WAITING FOR THE BACKEND"
          : hasImages
            ? "READY — RUN THE QUERY"
            : "FETCH OR UPLOAD IMAGERY TO BEGIN"}
      </p>
      <div className="mt-6 space-y-2.5">
        {steps.map(([k, v], i) => (
          <div key={k} className="flex gap-4">
            <span className="mono w-5 shrink-0 text-[10.5px] text-[var(--color-mute)]">
              {String(i + 1).padStart(2, "0")}
            </span>
            <span className="mono w-[92px] shrink-0 text-[11px] text-[var(--color-dim)]">{k}</span>
            <span className="text-[12.5px] leading-[1.5] text-[var(--color-mute)]">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
