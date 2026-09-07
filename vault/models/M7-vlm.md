---
model_id: "M7"
name: "vlm"
version: "0.1.0-untrained"
runtime: "llamacpp"
n_images: 1
trainable_on: "kaggle-p100 ~8-16h (own session)"
trained: false
tags: ["satquery/model"]
---

# M7 — `vlm`

**Tasks.** synthesis, vqa, caption
**Modalities.** any
**Images required.** 1
**Trainable params.** 2000.0M
**Training data.** [[BigEarthNet.txt-instruction-mix]]
**Compute.** kaggle-p100 ~8-16h (own session)

## Measured metrics

_Not trained yet — no metrics. Never put a guessed number here._

## Notes

Qwen3.5-2B-VL + QLoRA (r=16, a=32, NF4). fp16 NOT bf16 -- P100 is sm_60 and T4 sm_75; bf16 needs Ampere sm_80+ and will crash. Served via llama.cpp llama-server --mmproj (Ollama cannot load a separate mmproj for custom fine-tunes: ollama#14730, ollama#9967). Never sees raw imagery alone when specialists are available.
