# HarmoGATE

Research source code for the HarmoGATE model and the experiments reported in
the revised manuscript. The archive is intentionally limited to model-related
source files: the model definitions, data encoding and split preparation,
training, checkpoint loading, and prediction evaluation.

Public data files, generated splits, checkpoints, predictions, figures, logs,
and manuscript files are not included in this source archive. Dataset access,
local path configuration, and environment setup are required before training
or evaluation.

## Files

- `src/harmogate_model.py`: current model definition and detailed controls.
- `src/harmogate_model_legacy.py`: architecture used by the original seed-42
  checkpoints retained in the revised comparisons.
- `src/dataset.py`: SMILES, protein-sequence, and molecular-graph encoding.
- `src/train_harmogate.py`: joint, single-task, sensitivity, and ablation runs.
- `src/prepare_splits.py`: random, drug-disjoint, and target-disjoint splitting.
- `src/load_trained_model.py`: checkpoint compatibility loader.
- `src/evaluate_saved_predictions.py`: metric recomputation from saved arrays.

## Environment

The versions used for the revision experiments are listed in
`environment/requirements-harmogate.txt`. GPU execution additionally depends
on a PyTorch/CUDA build compatible with the local driver.

## Data layout

Training expects externally prepared CSV files at:

```text
data/<dataset>/<split>/train.csv
data/<dataset>/<split>/val.csv
data/<dataset>/<split>/test.csv
```

Each CSV requires `smiles`, `protein_sequence`, and `label` columns. The
`--data-root` argument may point to a different parent directory.

## Training controls

`train_harmogate.py --help` lists the dataset, split, split seed, training seed,
threshold, consistency weight, label weighting, and experiment controls. The
implemented experiment names are `full`, `dta_only`, `dti_only`,
`without_graph`, `without_cross_attention`, `equal_fusion`,
`global_attention_fusion`, `pure_global_attention_fusion`,
`without_modal_consistency`, `without_residual_mlp`,
`without_interaction_branch`, and `without_transformer`.

The script writes checkpoints and evaluation artifacts to the specified
revision output directory. It does not download or redistribute Davis or KIBA.

## Checkpoints

The two trained-model archives contain a `MODEL_MANIFEST.json` file. Its
`loader_variant` field identifies the compatible loader configuration for each
checkpoint. For example:

```text
python src/load_trained_model.py --checkpoint <path-to-pth> --variant full
```

The loader accepts both the current controls and the original seed-42
architectures and rejects incompatible state dictionaries.

## Saved predictions

Metrics can be recomputed from a directory containing the four saved NumPy
arrays (`y_true_dti`, `y_pred_dti`, `y_true_dta`, and `y_pred_dta`):

```text
python src/evaluate_saved_predictions.py <output-directory> --run-id <N>
```
