---
model_id: "M5b"
name: "change_vqa"
version: "0.1.0-untrained"
runtime: "onnx"
n_images: 2
trainable_on: "kaggle-p100 ~1h (shared trunk with M5a)"
trained: false
tags: ["satquery/model"]
---

# M5b — `change_vqa`

**Tasks.** change_vqa
**Modalities.** optical, sar
**Images required.** 2
**Trainable params.** 12.0M
**Training data.** [[CDVQA]]
**Compute.** kaggle-p100 ~1h (shared trunk with M5a)

## Measured metrics

_Not trained yet — no metrics. Never put a guessed number here._

## Notes

Shares M5a's siamese trunk; mask supervision improves the VQA head and it is one model to serve instead of two.
