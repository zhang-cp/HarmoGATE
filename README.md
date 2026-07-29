# HarmoGATE

Source code accompanying the HarmoGATE manuscript. This archive contains the
model definition, training and evaluation code, Davis/KIBA split-generation
code, and a checkpoint loader.

Public datasets and generated artifacts, including CSV splits, predictions,
metrics, and logs, are not redistributed.

## Environment

Install the dependencies listed in:

```text
environment/requirements-harmogate.txt
```

## Data preparation

`prepare_splits.py` creates `random`, `cold_drug`, and `cold_target`
train/validation/test CSV files. Training reads the selected split from
`data/<dataset>/<split>` by default.

## Training

The dataset, split, data root, threshold, seed, epoch count, batch size, and
ablation setting can be selected with the corresponding `HARMOGATE_*`
environment variables documented in `src/train_harmogate.py`.

## Loading a trained model

Extract a trained-model archive next to this source directory, then run:

```text
python src/load_trained_model.py --checkpoint <path-to-pth> --variant full
```

Available variants are `full`, `without_graph`, `without_cross_attention`,
`without_gated_fusion`, `without_modal_consistency`, and
`without_multitask_dti`.

The loader checks that each checkpoint is compatible with the released model
definition. Earlier full-model checkpoints do not contain the
`simple_fusion_norm` parameters introduced for the later
`without_gated_fusion` ablation. Those parameters are not used when gated
fusion is enabled; the loader permits only this known, functionally irrelevant
difference and rejects every other missing or unexpected parameter.

Saved predictions can be evaluated with:

```text
python src/evaluate_saved_predictions.py <output-directory> --run-id <N>
```
