
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


PREDICTION_STEMS = ("y_true_dti", "y_pred_dti", "y_true_dta", "y_pred_dta")


def resolve_prediction_files(result_dir, run_id=None):
    if run_id is not None:
        files = {
            stem: result_dir / f"{stem}_run{run_id}.npy"
            for stem in PREDICTION_STEMS
        }
        missing = [path.name for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete prediction files for run {run_id}: {', '.join(missing)}"
            )
        return files

    unsuffixed = {stem: result_dir / f"{stem}.npy" for stem in PREDICTION_STEMS}
    if all(path.is_file() for path in unsuffixed.values()):
        return unsuffixed

    complete_runs = []
    for path in result_dir.glob("y_true_dti_run*.npy"):
        suffix = path.stem.removeprefix("y_true_dti_run")
        if suffix.isdigit():
            candidate = {
                stem: result_dir / f"{stem}_run{suffix}.npy"
                for stem in PREDICTION_STEMS
            }
            if all(item.is_file() for item in candidate.values()):
                complete_runs.append((int(suffix), candidate))

    if len(complete_runs) == 1:
        return complete_runs[0][1]
    if len(complete_runs) > 1:
        available = ", ".join(str(run) for run, _ in sorted(complete_runs))
        raise ValueError(
            f"Multiple complete runs found ({available}); select one with --run-id."
        )
    raise FileNotFoundError(
        "No complete HarmoGATE prediction set was found in the result directory."
    )


def concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    left = y_true.reshape(-1, 1)
    right = y_true.reshape(1, -1)
    comparable = left < right
    predicted_left = y_pred.reshape(-1, 1)
    predicted_right = y_pred.reshape(1, -1)
    total = comparable.sum()
    concordant = ((predicted_left < predicted_right) & comparable).sum()
    tied = ((predicted_left == predicted_right) & comparable).sum()
    return float((concordant + 0.5 * tied) / total) if total else 0.5


def main():
    parser = argparse.ArgumentParser(description="Recompute metrics from saved HarmoGATE predictions.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    args = parser.parse_args()

    files = resolve_prediction_files(args.result_dir, args.run_id)
    yt_dti = np.load(files["y_true_dti"])
    yp_dti = np.load(files["y_pred_dti"])
    yt_dta = np.load(files["y_true_dta"])
    yp_dta = np.load(files["y_pred_dta"])
    predicted_class = (yp_dti >= args.classification_threshold).astype(int)
    mse = mean_squared_error(yt_dta, yp_dta)
    metrics = {
        "AUC": roc_auc_score(yt_dti, yp_dti),
        "AUPR": average_precision_score(yt_dti, yp_dti),
        "F1": f1_score(yt_dti, predicted_class, zero_division=0),
        "Accuracy": accuracy_score(yt_dti, predicted_class),
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(yt_dta, yp_dta),
        "R2": r2_score(yt_dta, yp_dta),
        "CI": concordance_index(yt_dta, yp_dta),
        "PCC": pearsonr(yt_dta, yp_dta)[0],
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
