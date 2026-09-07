---
model_id: "M6"
name: "fusion"
version: "0.1.0-untrained"
runtime: "onnx"
n_images: 2
trainable_on: "kaggle-p100 ~3h"
trained: false
tags: ["satquery/model"]
---

# M6 — `fusion`

**Tasks.** fusion
**Modalities.** optical, sar
**Images required.** 2
**Trainable params.** 48.0M
**Training data.** [[BigEarthNet-v2-co-registered-S1+S2,-BEN-19-multilabel]]
**Compute.** kaggle-p100 ~3h

## Measured metrics

_Not trained yet — no metrics. Never put a guessed number here._

## Notes

S2 (12-band) + S1 (VV/VH) branches -> cross-attention -> BEN-19 multilabel. Init from BIFOLD resnet50-s2 / resnet101-s1. MUST normalise per-sensor from metadata -- ISRO evaluates on Cartosat-2S + RISAT, not Sentinel. Never hardcode Sentinel stats.
