"use client";

import { useCallback, useEffect, useState } from "react";
import { Nav } from "@/components/Chrome";
import { Footer } from "@/components/Footer";

/**
 * A live status board for the FastAPI service, not the console itself.
 *
 * The full query console is the next piece of work. What this page does today
 * is prove the wiring end to end: it calls the same /health and /models
 * endpoints the console will use, through the Next rewrite in next.config.ts,
 * and shows exactly what came back. If this page is green, the backend is up
 * and every registered model has its weights on disk.
 */

type Health = {
  status: string;
  models_registered: number;
  models_trained: string[];
  mandatory_coverage: Record<string, string[]>;
};

type Model = {
  id: string;
  name: string;
  version: string;
  tasks: string[];
  modalities: string[];
  n_images: number;
  runtime: string;
  params_m: number | null;
  trained: boolean;
  notes: string;
};

type State =
  | { kind: "loading" }
  | { kind: "down"; error: string }
  | { kind: "up"; health: Health; models: Model[] };

export default function ConsolePage() {
  const [state, setState] = useState<State>({ kind: "loading" });

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    try {
      const [h, m] = await Promise.all([
        fetch("/api/health", { cache: "no-store" }),
        fetch("/api/models", { cache: "no-store" }),
      ]);
      if (!h.ok || !m.ok) throw new Error(`HTTP ${h.status} / ${m.status}`);
      setState({ kind: "up", health: await h.json(), models: await m.json() });
    } catch (e) {
      setState({ kind: "down", error: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <>
      <Nav />
      <main id="main" className="mx-auto max-w-[1240px] px-5 pb-24 pt-[128px] sm:px-8">
        <p className="eyebrow">Console</p>
        <h1 className="display mt-5 max-w-[20ch] text-[clamp(1.8rem,4vw,2.8rem)] text-white">
          Backend status.
        </h1>
        <p className="mt-5 max-w-[64ch] text-[15px] leading-[1.7] text-[var(--color-mute)]">
          The query console is the next piece of work. This page talks to the same
          FastAPI service it will use — <span className="mono">/health</span> and{" "}
          <span className="mono">/models</span> — so the wiring is verifiable now.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-4">
          <StatusPill state={state} />
          <button
            type="button"
            onClick={() => void load()}
            className="mono rounded-full border border-[var(--color-line)] px-4 py-[7px] text-[11.5px] tracking-[0.12em] text-[var(--color-dim)] transition-colors hover:border-white/25 hover:text-white"
          >
            REFRESH
          </button>
        </div>

        {state.kind === "down" && (
          <div className="panel mt-8 rounded-lg p-6">
            <p className="text-[14.5px] text-white">The service is not reachable.</p>
            <p className="mono mt-2 text-[12px] text-[var(--color-signal)]">{state.error}</p>
            <p className="mt-4 text-[13.5px] leading-[1.7] text-[var(--color-mute)]">
              Start it from the repository root:
            </p>
            <pre className="mono mt-3 overflow-x-auto rounded bg-white/[0.04] px-4 py-3 text-[12px] text-[var(--color-dim)]">
              uvicorn serve.api:app --reload --port 8000
            </pre>
          </div>
        )}

        {state.kind === "up" && (
          <>
            <div className="mt-10 grid gap-px overflow-hidden rounded-lg border border-[var(--color-line)] bg-[var(--color-line)] sm:grid-cols-3">
              <Tile label="REGISTERED" value={String(state.health.models_registered)} />
              <Tile
                label="WEIGHTS ON DISK"
                value={`${state.health.models_trained.length}/${state.health.models_registered}`}
              />
              <Tile
                label="MANDATORY COVERAGE"
                value={`${Object.keys(state.health.mandatory_coverage).length}/4`}
              />
            </div>

            <div className="mt-10 overflow-hidden rounded-lg border border-[var(--color-line)]">
              {state.models.map((m) => (
                <div
                  key={m.id}
                  className="grid grid-cols-1 gap-2 border-b border-[var(--color-line)] px-5 py-4 last:border-0 sm:grid-cols-[60px_1fr_1fr_100px_90px] sm:items-center sm:gap-4"
                >
                  <span className="mono text-[13px] font-medium text-white">{m.id}</span>
                  <span className="mono text-[12.5px] text-[var(--color-brand-soft)]">
                    {m.name}
                    <span className="ml-2 text-[var(--color-mute)]">v{m.version}</span>
                  </span>
                  <span className="mono text-[11.5px] text-[var(--color-mute)]">
                    {m.tasks.join(" · ")}
                  </span>
                  <span className="mono text-[11.5px] text-[var(--color-mute)]">
                    {m.runtime}
                    {m.params_m ? ` · ${m.params_m}M` : ""}
                  </span>
                  <span
                    className={`mono text-[10px] tracking-[0.14em] ${
                      m.trained ? "text-[var(--color-verd)]" : "text-[var(--color-signal)]"
                    }`}
                  >
                    {m.trained ? "TRAINED" : "PENDING"}
                  </span>
                </div>
              ))}
            </div>

            <div className="mt-10 grid gap-px overflow-hidden rounded-lg border border-[var(--color-line)] bg-[var(--color-line)] sm:grid-cols-2 lg:grid-cols-4">
              {Object.entries(state.health.mandatory_coverage).map(([req, ids]) => (
                <div key={req} className="bg-black p-5">
                  <p className="text-[13px] leading-snug text-[var(--color-dim)]">{req}</p>
                  <p className="mono mt-2.5 text-[12px] text-[var(--color-verd)]">
                    {ids.join(", ")}
                  </p>
                </div>
              ))}
            </div>
          </>
        )}
      </main>
      <Footer />
    </>
  );
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-black px-6 py-6">
      <p className="text-[26px] font-medium tabular-nums tracking-tight text-white">
        {value}
      </p>
      <p className="mono mt-2 text-[10.5px] tracking-[0.14em] text-[var(--color-mute)]">
        {label}
      </p>
    </div>
  );
}

function StatusPill({ state }: { state: State }) {
  const [dot, text] =
    state.kind === "up"
      ? ["bg-[var(--color-verd)]", "SERVICE UP"]
      : state.kind === "loading"
        ? ["bg-[var(--color-mute)]", "CHECKING"]
        : ["bg-[var(--color-signal)]", "SERVICE DOWN"];
  return (
    <span className="mono inline-flex items-center gap-2.5 rounded-full border border-[var(--color-line)] px-4 py-[7px] text-[11.5px] tracking-[0.12em] text-[var(--color-dim)]">
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {text}
    </span>
  );
}
