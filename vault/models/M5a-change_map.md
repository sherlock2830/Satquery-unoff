---
model_id: "M5a"
name: "change_map"
version: "0.1.0-untrained"
runtime: "onnx"
n_images: 2
trainable_on: "kaggle-p100 ~2h (shared trunk with M5b)"
trained: false
tags: ["satquery/model"]
---

# M5a — `change_map`

**Tasks.** change_map
**Modalities.** optical, sar
**Images required.** 2
**Trainable params.** 31.0M
**Training data.** [[LEVIR-CD]]
**Compute.** kaggle-p100 ~2h (shared trunk with M5b)

## Measured metrics

_Not trained yet — no metrics. Never put a guessed number here._

## Notes

Siamese M1 -> feature differencing -> light U-Net decoder -> binary mask. Produces the spatial evidence overlay.
