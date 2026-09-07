---
model_id: "M1"
name: "rsclip"
version: "0.1.0-untrained"
runtime: "onnx"
n_images: 1
trainable_on: "kaggle-p100 ~4h"
trained: false
tags: ["satquery/model"]
---

# M1 — `rsclip`

**Tasks.** retrieval
**Modalities.** optical, sar
**Images required.** 1
**Trainable params.** 151.0M
**Training data.** [[BigEarthNet.txt-captions-x-reBEN-patches]]
**Compute.** kaggle-p100 ~4h

## Measured metrics

_Not trained yet — no metrics. Never put a guessed number here._

## Notes

Domain-adaptation core. ViT-B/32 with patch-embed inflated 3->14 channels (12 S2 bands + VV/VH). Its frozen image tower is the backbone of M2, M3, M4 and M6 -- train first.
