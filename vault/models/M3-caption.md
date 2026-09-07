---
model_id: "M3"
name: "caption"
version: "0.1.0-untrained"
runtime: "onnx"
n_images: 1
trainable_on: "kaggle-p100 ~2h"
trained: false
tags: ["satquery/model"]
---

# M3 — `caption`

**Tasks.** caption
**Modalities.** optical, sar
**Images required.** 1
**Trainable params.** 25.0M
**Training data.** [[BigEarthNet.txt-geographically-anchored-captions]]
**Compute.** kaggle-p100 ~2h

## Measured metrics

_Not trained yet — no metrics. Never put a guessed number here._

## Notes

Prefix-tuned: M1 embedding -> 10 soft prompt tokens -> small causal decoder. ~50ms on CPU vs ~4s for M7, and a working fallback if the VLM fails on demo day.
