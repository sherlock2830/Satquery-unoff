# SatQuery AI — core

**SIH 2026 · PS 26167 · ISRO/SAC** — agentic vision-language assistant for
multimodal remote sensing.

Final product. Supersedes `satquery/`, `sonar/` and `vqa/`.
Start with **[docs/MASTER_PLAN.md](docs/MASTER_PLAN.md)**.

---

## Status

| | |
|---|---|
| Agent graph (LangGraph 1.2) | ✅ running end-to-end |
| Model registry, 8 entries | ✅ |
| Execution trace → Obsidian vault | ✅ |
| FastAPI service | ✅ |
| Map AOI console (Leaflet + live Sentinel fetch) | ✅ |
| **IndiaSat dataset** | ✅ **985 patches · 26,595 raster images · 8,962 annotations · 8 regions · 100% India** |
| **M1–M6 trained on CPU** | ✅ **7/8 registry entries — all beat their baselines** |
| M7 (Qwen3.5-2B-VL QLoRA) | ⬜ needs the Kaggle GPU (bitsandbytes is CUDA-only) |
| M1 Kaggle notebook (full ViT-B/32) | ✅ written, smoke-tested on CPU |

| model | metric | result | baseline |
|---|---|---:|---:|
| M6 fusion | mAP | **0.9150** | 0.3424 prior |
| M3 caption | next-token acc | **0.9010** | 0.0743 mode |
| M5b change-VQA | accuracy | **0.7682** | 0.4702 majority |
| M2 vqa | overall acc | **0.7587** | 0.5749 majority |
| M5a change map | IoU | **0.5090** | — |
| M4 ground | mean IoU | **0.3082** | 0.1983 mean box |
| M1 rsclip | R@1 (n=151) | **0.1391** | 0.0066 chance |

Full numbers, the optical–SAR cloud ablation, and the caveats that matter:
**[docs/RESULTS.md](docs/RESULTS.md)**.

Stubbed models return `stub: true` and are stamped `STUB` in the trace, so a
development screenshot can never be mistaken for a measured result.

## Quick start

```bash
pip install langgraph fastapi uvicorn onnxruntime pillow numpy datasets huggingface_hub rasterio
```

```bash
python scripts/fetch_worldcover.py       # 7 label tiles, ~940 MB
python scripts/build_india_dataset.py    # build IndiaSat from S2+S1+WorldCover
python scripts/smoke_test_m1.py          # CPU check of the M1 training path
python scripts/verify_onnx_export.py     # numerical ONNX fidelity check
```

```bash
python -m agent.registry     # registry + mandatory-coverage check
python -m agent.router       # routing on the PS representative queries
python -m agent.graph        # full pipeline, writes vault notes
python -m agent.trace        # regenerate vault model notes
```

Train the CPU-trainable specialists on IndiaSat:

```bash
python -m models.train_all          # M1, M3, M4, M5, M6 — reports in models/reports/
```

Serve it, then open the test console:

```bash
uvicorn serve.api:app --reload --port 8000
```

Then visit **http://127.0.0.1:8000** — upload one or two images, ask a question,
and see the answer, the models the agent selected, per-step latencies and
confidences, and a downloadable Markdown report.

API: `GET /` · `GET /health` · `GET /models` · `POST /upload` · `POST /query` · `GET /runs`

The Next.js dashboard in `../satquery/` can talk to the same API — no frontend
rewrite needed.

## Layout

```
docs/MASTER_PLAN.md    architecture, training recipes, hosting decision
docs/RESULTS.md        measured metrics + baselines + honest caveats
docs/OBSIDIAN_GUIDE.md setup, graph view, Dataview queries, demo script
serve/aoi.py           live Sentinel fetch for a map-drawn AOI
agent/torch_runtime.py serving adapters for the CPU-trained specialists
models/backbone.py     shared 14-channel CNN encoder + 5 task heads
models/train_all.py    trains M1/M3/M4/M5/M6 on CPU with baselines
serve/static/index.html  the test console
agent/registry.py      declarative model registry — the router's only source
agent/router.py        hybrid rule + LLM task routing, with refusal reasons
agent/graph.py         LangGraph state machine
agent/trace.py         execution trace → JSON + Obsidian markdown
agent/tools.py         per-runtime model adapters (ONNX / llama.cpp)
serve/api.py           FastAPI
scripts/               dataset fetchers
vault/                 Obsidian vault — run notes, model notes, AOIs
models/weights/        ONNX + GGUF artefacts (gitignored)
```

## IndiaSat — a 100% Indian corpus

BigEarthNet.txt contains **zero** Indian data (10 European countries; its
latitude/longitude ranges do not overlap India at all). Root cause: its labels
come from CORINE Land Cover, a Europe-only product. Swapping CORINE for **ESA
WorldCover** (10 m, global, CC-BY-4.0) reproduces the recipe over India.

489 patches across 8 regions spanning tropical monsoon, arid desert, savannah,
humid subtropical and montane — **every Köppen zone absent from BigEarthNet**.
Sentinel-2 (12 band) + Sentinel-1 (VV/VH) + WorldCover labels, co-registered,
all from keyless public sources. Annotations use the BigEarthNet.txt schema, so
the two corpora concatenate into one training mix.

## Two decisions worth knowing up front

**Training runs on Kaggle, not here.** This laptop has no NVIDIA GPU, so QLoRA
is impossible locally (`bitsandbytes` is CUDA-only). Kaggle's free tier gives 30
GPU-hours/week on a P100/T4. Use **fp16, never bf16** — P100 is sm_60 and T4 is
sm_75; bf16 needs Ampere sm_80+ and will crash on the first step.

**Serving uses llama.cpp, not Ollama.** Ollama cannot load a separate `mmproj`
vision projector for a custom fine-tuned VLM
([ollama#14730](https://github.com/ollama/ollama/issues/14730),
[ollama#9967](https://github.com/ollama/ollama/issues/9967)) — and it fails by
silently dropping vision, which is the worst possible failure mode. `llama-server
--mmproj` works. Full reasoning in §4 of the master plan.

## The vault

Every run writes a wikilinked markdown note to `vault/`. Open that folder as an
Obsidian vault and the graph view renders the live provenance network — every
query, every model, every AOI. Obsidian only ever *reads* the folder; it is
never in the request path, and everything works without it installed.
