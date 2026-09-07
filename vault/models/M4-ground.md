---
model_id: "M4"
name: "ground"
version: "0.1.0-untrained"
runtime: "onnx"
n_images: 1
trainable_on: "kaggle-p100 ~3h"
trained: false
tags: ["satquery/model"]
---

# M4 — `ground`

**Tasks.** grounding
**Modalities.** optical, sar
**Images required.** 1
**Trainable params.** 38.0M
**Training data.** [[BigEarthNet.txt-referring-expression-+-bbox-annotations]]
**Compute.** kaggle-p100 ~3h

## Measured metrics

_Not trained yet — no metrics. Never put a guessed number here._

## Notes

M1 patch grid + text tower -> 2-layer cross-attention -> box head. L1 + generalised IoU loss. Answers 'highlight the water body'.
