import os
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import pearsonr
from tqdm import tqdm

from dataset import (
    UniversalDTIDataset,
    dti_collate_fn,
    DRUG_DICT,
    PROT_DICT,
)
from harmogate_model import HarmoGATE


# =========================
# 基础配置
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HarmoGATE and the reported single-task or ablation controls."
    )
    parser.add_argument("--dataset", choices=["davis", "kiba"], default="kiba")
    parser.add_argument(
        "--split", choices=["random", "cold_drug", "cold_target"], default="random"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used to create the supplied split; tracked separately from training seed.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--dti-threshold", type=float, default=None)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument(
        "--dta-label-weighting",
        choices=["submitted", "none"],
        default="submitted",
        help="Use the submitted Equation 18 weighting or an unweighted DTA sensitivity control.",
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "full",
            "dta_only",
            "dti_only",
            "without_graph",
            "without_cross_attention",
            "equal_fusion",
            "global_attention_fusion",
            "pure_global_attention_fusion",
            "without_modal_consistency",
            "without_residual_mlp",
            "without_interaction_branch",
            "without_transformer",
        ],
        default="full",
    )
    parser.add_argument(
        "--data-root",
        default=str(SCRIPT_DIR.parent / "data"),
        help="Root containing <dataset>/<split>/{train,val,test}.csv.",
    )
    parser.add_argument(
        "--revision-root",
        default=None,
        help="Write an isolated revision run below this directory.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate data and execute one training batch without writing a formal run.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Resume an incomplete matching run; completed runs remain protected.",
    )
    return parser.parse_args()


ARGS = parse_args()
SEED = ARGS.seed
EPOCHS = ARGS.epochs
BATCH_SIZE = ARGS.batch_size
LR = ARGS.lr
WEIGHT_DECAY = ARGS.weight_decay

DATASET = ARGS.dataset
SPLIT = ARGS.split
DATA_ROOT = str((Path(ARGS.data_root) / DATASET / SPLIT).resolve())

DTA_THRESHOLD = ARGS.dti_threshold
if DTA_THRESHOLD is None:
    DTA_THRESHOLD = 7.0 if DATASET == "davis" else 12.1

# 修改：君臣佐使默认关闭。公共数据集、没有君臣佐使字段的中药数据集都保持 False。
# 如果以后 jc_code 有明确含义，例如 0/1/2/3 对应君臣佐使，再改成 True。
USE_JCZS = False

# 修改：本轮优先补 DTA 数值回归精度，暂时关闭 uncertainty loss。
# 原因：uncertainty loss 会让总 loss 变负，且可能更偏向排序/分类；固定权重更利于稳定压低 RMSE/MAE。
USE_UNCERTAINTY_LOSS = False

# 修改：固定 loss 权重备用。只有 USE_UNCERTAINTY_LOSS = False 时生效。
DTI_LOSS_WEIGHT = 0.1
DTA_LOSS_WEIGHT = 1.0

# 修改：DTA 回归损失使用 SmoothL1 + MSE 混合。
# SmoothL1 降低异常亲和力点对训练的冲击，MSE 继续约束整体数值误差。
DTA_SMOOTH_L1_WEIGHT = 0.7
DTA_MSE_WEIGHT = 0.3

# 修改：模态一致性约束默认开启，用于约束 SMILES 分支和图分支的药物表示。
USE_MODAL_CONSISTENCY = True
MODAL_CONSISTENCY_WEIGHT = ARGS.consistency_weight

# 修改：监督式对比学习先默认关闭。需要做增强实验时再改成 True。
USE_CONTRASTIVE = False
CONTRASTIVE_WEIGHT = 0.02

# 消融与单任务实验配置。默认 full 保持历史完整模型的计算路径。
ABLATION_NAME = ARGS.experiment
USE_GRAPH_BRANCH = True
USE_CROSS_ATTENTION = True
USE_GATED_FUSION = True
USE_SEQUENCE_TRANSFORMER = True
USE_INTERACTION_BRANCH = True
USE_FUSION_RESIDUAL = True
FUSION_MODE = "gated"
TASK_MODE = "joint"

if ABLATION_NAME == "full":
    pass
elif ABLATION_NAME == "dta_only":
    TASK_MODE = "dta_only"
    DTI_LOSS_WEIGHT = 0.0
elif ABLATION_NAME == "dti_only":
    TASK_MODE = "dti_only"
    DTI_LOSS_WEIGHT = 1.0
    DTA_LOSS_WEIGHT = 0.0
elif ABLATION_NAME == "without_graph":
    USE_GRAPH_BRANCH = False
    USE_MODAL_CONSISTENCY = False
elif ABLATION_NAME == "without_cross_attention":
    USE_CROSS_ATTENTION = False
elif ABLATION_NAME == "equal_fusion":
    USE_GATED_FUSION = False
    FUSION_MODE = "equal"
elif ABLATION_NAME == "global_attention_fusion":
    USE_GATED_FUSION = False
    FUSION_MODE = "global_attention"
elif ABLATION_NAME == "pure_global_attention_fusion":
    USE_GATED_FUSION = False
    FUSION_MODE = "global_attention_residual"
elif ABLATION_NAME == "without_modal_consistency":
    USE_MODAL_CONSISTENCY = False
elif ABLATION_NAME == "without_residual_mlp":
    USE_FUSION_RESIDUAL = False
elif ABLATION_NAME == "without_interaction_branch":
    USE_INTERACTION_BRANCH = False
    USE_CROSS_ATTENTION = False
elif ABLATION_NAME == "without_transformer":
    USE_SEQUENCE_TRANSFORMER = False
else:
    raise ValueError(f"Unknown ABLATION_NAME: {ABLATION_NAME}")

EARLY_STOP_PATIENCE = ARGS.patience

# 修改：早停和 best model 保存指标。默认仍然使用 val_ci，更适合 DTA 回归任务。
# 如果更重视 DTI 分类，可以改成 "val_auc"。
BEST_METRIC_NAME = "val_auc" if TASK_MODE == "dti_only" else "val_dta_balanced"

# 修改：训练集样本较大时完整 CI 需要 O(N^2) 两两比较，KIBA/BindingDB 会非常慢或占用大量内存。
# 训练阶段只抽样估计 train_ci；验证集和测试集仍然完整计算 CI。
TRAIN_CI_SAMPLE_SIZE = 3000

# 新增：只评估已有 run，不重新训练。
# 当前先评估 run7；评估完成后请改回 None，避免下次训练被跳过。
EVAL_ONLY_RUN_ID = None

DEVICE = ARGS.device or ("cuda" if torch.cuda.is_available() else "cpu")

REVISION_RUN_DIR = None
if ARGS.revision_root:
    experiment_slug = ABLATION_NAME
    if abs(MODAL_CONSISTENCY_WEIGHT - 0.05) > 1e-12:
        experiment_slug += f"__consistency_{MODAL_CONSISTENCY_WEIGHT:g}"
    default_threshold = 7.0 if DATASET == "davis" else 12.1
    if abs(DTA_THRESHOLD - default_threshold) > 1e-12:
        experiment_slug += f"__threshold_{DTA_THRESHOLD:g}"
    if ARGS.dta_label_weighting != "submitted":
        experiment_slug += f"__dta_weighting_{ARGS.dta_label_weighting}"
    split_slug = SPLIT
    if SPLIT != "random" and ARGS.split_seed != 42:
        split_slug += f"__split_seed_{ARGS.split_seed}"
    REVISION_RUN_DIR = (
        Path(ARGS.revision_root).resolve()
        / DATASET
        / split_slug
        / experiment_slug
        / f"seed_{SEED}"
    )
    if REVISION_RUN_DIR.exists():
        manifest_path = REVISION_RUN_DIR / "run_manifest.json"
        manifest_complete = False
        if manifest_path.is_file():
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    existing_manifest = json.load(handle)
                manifest_complete = (
                    existing_manifest.get("status") == "complete"
                    and existing_manifest.get("test_evaluations") == 1
                )
            except (OSError, ValueError, TypeError):
                manifest_complete = False
        if manifest_complete:
            raise FileExistsError(f"Completed revision run is protected: {REVISION_RUN_DIR}")
        if not ARGS.allow_existing:
            raise FileExistsError(
                f"Revision output already exists: {REVISION_RUN_DIR}. "
                "Use --allow-existing only to resume an incomplete matching run."
            )
    REVISION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(REVISION_RUN_DIR)

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("output", exist_ok=True)


# =========================
# 随机种子
# 修改：加入随机种子，减少每次训练结果波动
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================
# run_id 与续跑机制
# 修改：保留并升级续跑机制，使用 latest_checkpoint_x.pth 保存完整状态
# =========================
def get_run_ids():
    files = [f for f in os.listdir("logs") if f.startswith("train_log_") and f.endswith(".csv")]
    ids = []
    for f in files:
        try:
            ids.append(int(f.replace("train_log_", "").replace(".csv", "")))
        except ValueError:
            pass
    return ids


def get_new_run_id():
    ids = get_run_ids()
    return max(ids, default=0) + 1


def get_last_run_id():
    ids = get_run_ids()
    return max(ids, default=0)


def is_checkpoint_compatible(ckpt):
    """只允许同一数据集和同一关键配置的 checkpoint 自动续跑。"""
    required = {
        "seed": SEED,
        "split_seed": ARGS.split_seed,
        "dataset": DATASET,
        "data_root": DATA_ROOT,
        "dta_threshold": DTA_THRESHOLD,
        "use_jczs": USE_JCZS,
        "use_uncertainty_loss": USE_UNCERTAINTY_LOSS,
        "best_metric_name": BEST_METRIC_NAME,
        "ablation_name": ABLATION_NAME,
        "use_graph_branch": USE_GRAPH_BRANCH,
        "use_cross_attention": USE_CROSS_ATTENTION,
        "use_gated_fusion": USE_GATED_FUSION,
        "use_sequence_transformer": USE_SEQUENCE_TRANSFORMER,
        "use_interaction_branch": USE_INTERACTION_BRANCH,
        "use_fusion_residual": USE_FUSION_RESIDUAL,
        "fusion_mode": FUSION_MODE,
        "task_mode": TASK_MODE,
        "modal_consistency_weight": MODAL_CONSISTENCY_WEIGHT,
        "dti_loss_weight": DTI_LOSS_WEIGHT,
        "dta_loss_weight": DTA_LOSS_WEIGHT,
        "dta_label_weighting": ARGS.dta_label_weighting,
    }

    for key, expected in required.items():
        if key not in ckpt:
            print(f"Skip resume: checkpoint missing {key}, run is not config-safe.")
            return False
        actual = ckpt.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-8:
                print(f"Skip resume: checkpoint {key}={actual} != current {expected}.")
                return False
        elif actual != expected:
            print(f"Skip resume: checkpoint {key}={actual} != current {expected}.")
            return False

    return True


def should_resume_run(last_id):
    if last_id <= 0:
        return False

    last_log = f"logs/train_log_{last_id}.csv"
    last_ckpt = f"checkpoints/latest_checkpoint_{last_id}.pth"

    if not os.path.exists(last_log) or not os.path.exists(last_ckpt):
        return False

    try:
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        if not is_checkpoint_compatible(ckpt):
            return False

        # A finished training loop may still need its one final test pass if the
        # process stopped while evaluating or saving outputs.
        if ckpt.get("finished", False):
            return REVISION_RUN_DIR is not None

        df = pd.read_csv(last_log)
        if len(df) == 0:
            return False

        last_epoch = int(df["epoch"].max())
        total_epochs = int(df["total_epochs"].iloc[-1]) if "total_epochs" in df.columns else EPOCHS
        return last_epoch < total_epochs
    except Exception:
        return False


def find_resume_run_id():
    ids = sorted(get_run_ids(), reverse=True)
    for rid in ids:
        if should_resume_run(rid):
            return rid
    return None


if EVAL_ONLY_RUN_ID is not None:
    run_id = EVAL_ONLY_RUN_ID
    resume = False
else:
    resume_id = find_resume_run_id()
    resume = resume_id is not None

    if resume:
        run_id = resume_id
    else:
        run_id = get_new_run_id()

if REVISION_RUN_DIR is not None:
    existing_ids = get_run_ids()
    if existing_ids and not resume:
        raise RuntimeError(
            "Existing revision artifacts do not form a resumable matching run; "
            "inspect them instead of starting a second run in the same directory."
        )
    if run_id != 1:
        raise RuntimeError("A semantic revision directory may contain only run_id=1")


log_file = f"logs/train_log_{run_id}.csv"
best_model_file = f"checkpoints/best_model_{run_id}.pth"
latest_model_file = f"checkpoints/latest_model_{run_id}.pth"
latest_ckpt_file = f"checkpoints/latest_checkpoint_{run_id}.pth"

print("=" * 70)
print(f"Run ID: {run_id}")
print(f"Dataset: {DATASET}")
print(f"Split: {SPLIT}")
print(f"Device: {DEVICE}")
print(f"Use JCZS: {USE_JCZS}")
print(f"Use uncertainty loss: {USE_UNCERTAINTY_LOSS}")
print(f"Ablation: {ABLATION_NAME}")
print(
    "Ablation switches: "
    f"graph={USE_GRAPH_BRANCH}, cross_attn={USE_CROSS_ATTENTION}, "
    f"interaction={USE_INTERACTION_BRANCH}, fusion={FUSION_MODE}, "
    f"fusion_residual={USE_FUSION_RESIDUAL}, transformer={USE_SEQUENCE_TRANSFORMER}, "
    f"modal_consistency={USE_MODAL_CONSISTENCY}, "
    f"dti_loss_weight={DTI_LOSS_WEIGHT}, dta_loss_weight={DTA_LOSS_WEIGHT}"
)
print(f"Best metric: {BEST_METRIC_NAME}")
print(f"Resume: {resume}")
if REVISION_RUN_DIR is not None:
    print(f"Revision output: {REVISION_RUN_DIR}")
print("=" * 70)


# =========================
# 数据加载
# Use the released dataset and collate implementation.
# =========================
train_df = pd.read_csv(f"{DATA_ROOT}/train.csv")
val_df = pd.read_csv(f"{DATA_ROOT}/val.csv")
test_df = pd.read_csv(f"{DATA_ROOT}/test.csv")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_frames(frames):
    required = {"smiles", "protein_sequence"}
    pair_sets = {}
    for name, frame in frames.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv missing columns: {sorted(missing)}")
        label_name = next(
            (candidate for candidate in ["label", "Label", "affinity", "y"] if candidate in frame.columns),
            None,
        )
        if label_name is None:
            raise ValueError(f"{name}.csv has no affinity label column")
        labels = pd.to_numeric(frame[label_name], errors="coerce").to_numpy()
        if not np.isfinite(labels).all():
            raise ValueError(f"{name}.csv contains non-finite affinity labels")
        pairs = set(zip(frame["smiles"].astype(str), frame["protein_sequence"].astype(str)))
        if len(pairs) != len(frame):
            raise ValueError(f"{name}.csv contains duplicate drug-target pairs")
        pair_sets[name] = pairs

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = pair_sets[left] & pair_sets[right]
        if overlap:
            raise ValueError(f"Pair leakage between {left} and {right}: {len(overlap)}")

    if SPLIT == "cold_drug":
        entities = {
            name: set(frame["smiles"].astype(str)) for name, frame in frames.items()
        }
    elif SPLIT == "cold_target":
        entities = {
            name: set(frame["protein_sequence"].astype(str)) for name, frame in frames.items()
        }
    else:
        entities = None
    if entities is not None:
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = entities[left] & entities[right]
            if overlap:
                raise ValueError(
                    f"Entity leakage in {SPLIT} between {left} and {right}: {len(overlap)}"
                )


split_frames = {"train": train_df, "val": val_df, "test": test_df}
validate_split_frames(split_frames)

if REVISION_RUN_DIR is not None:
    split_paths = {name: Path(DATA_ROOT) / f"{name}.csv" for name in split_frames}
    run_config = {
        "status": "initialized",
        "model": "HarmoGATE",
        "dataset": DATASET,
        "split": SPLIT,
        "seed": SEED,
        "split_seed": ARGS.split_seed,
        "experiment": ABLATION_NAME,
        "task_mode": TASK_MODE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "selection_set": "validation",
        "selection_metric": BEST_METRIC_NAME,
        "test_evaluations": 0,
        "dti_affinity_threshold": DTA_THRESHOLD,
        "dti_probability_threshold": 0.5,
        "dti_loss_weight": DTI_LOSS_WEIGHT,
        "dta_loss_weight": DTA_LOSS_WEIGHT,
        "dta_label_weighting": ARGS.dta_label_weighting,
        "modal_consistency_enabled": USE_MODAL_CONSISTENCY,
        "modal_consistency_weight": MODAL_CONSISTENCY_WEIGHT,
        "use_graph_branch": USE_GRAPH_BRANCH,
        "use_cross_attention": USE_CROSS_ATTENTION,
        "use_interaction_branch": USE_INTERACTION_BRANCH,
        "fusion_mode": FUSION_MODE,
        "use_fusion_residual": USE_FUSION_RESIDUAL,
        "use_sequence_transformer": USE_SEQUENCE_TRANSFORMER,
        "device": DEVICE,
        "command": [sys.executable, str(SCRIPT_DIR / "train_harmogate.py"), *sys.argv[1:]],
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "source_files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in {
                "trainer": SCRIPT_DIR / "train_harmogate.py",
                "model": SCRIPT_DIR / "harmogate_model.py",
                "dataset": SCRIPT_DIR / "dataset.py",
            }.items()
        },
        "split_files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": int(len(split_frames[name])),
            }
            for name, path in split_paths.items()
        },
    }
    with Path("run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)


def get_label_column(df):
    for name in ["label", "Label", "affinity", "y"]:
        if name in df.columns:
            return name
    raise ValueError("No label column found. Expected one of: label, Label, affinity, y")


label_col = get_label_column(train_df)
train_labels = train_df[label_col].astype(float).values
train_dti_labels = (train_labels >= DTA_THRESHOLD).astype(float)

pos = train_dti_labels.sum()
neg = len(train_dti_labels) - pos
if pos > 0:
    pos_weight = torch.tensor([math.sqrt(neg / pos) * 0.7], dtype=torch.float32).to(DEVICE)
else:
    pos_weight = torch.tensor([1.0], dtype=torch.float32).to(DEVICE)

train_dataset = UniversalDTIDataset(
    train_df,
    dataset_type=DATASET,
    dta_threshold=DTA_THRESHOLD,
    use_jczs=USE_JCZS,
)

val_dataset = UniversalDTIDataset(
    val_df,
    dataset_type=DATASET,
    dta_threshold=DTA_THRESHOLD,
    use_jczs=USE_JCZS,
)

test_dataset = UniversalDTIDataset(
    test_df,
    dataset_type=DATASET,
    dta_threshold=DTA_THRESHOLD,
    use_jczs=USE_JCZS,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=dti_collate_fn,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=dti_collate_fn,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=dti_collate_fn,
    num_workers=0,
)


# =========================
# Loss 函数
# =========================
def dta_regression_loss(pred, true):
    mse_loss = (pred - true) ** 2
    smooth_l1_loss = F.smooth_l1_loss(pred, true, beta=0.5, reduction="none")

    weight = torch.ones_like(true)
    if ARGS.dta_label_weighting == "submitted":
        weight[true > 7.0] = 1.2
        weight[true > 8.0] = 1.5
        weight[true < 5.5] = 0.9

    loss = DTA_SMOOTH_L1_WEIGHT * smooth_l1_loss + DTA_MSE_WEIGHT * mse_loss
    return (loss * weight).mean()


class UncertaintyMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_dti = nn.Parameter(torch.zeros(1))
        self.log_var_dta = nn.Parameter(torch.zeros(1))

    def forward(self, loss_dti, loss_dta):
    # 新增：限制 log_var 范围，避免自动权重无限放大导致 loss 过度变负
        log_var_dti = torch.clamp(self.log_var_dti, min=-1.5, max=1.5)
        log_var_dta = torch.clamp(self.log_var_dta, min=-1.5, max=1.5)

        precision_dti = torch.exp(-log_var_dti)
        precision_dta = torch.exp(-log_var_dta)

        return (
           precision_dti * loss_dti + log_var_dti
           + precision_dta * loss_dta + log_var_dta
        )



def compute_multitask_loss(loss_dti, loss_dta, uncertainty_criterion=None):
    # 修改：这里是自动权重和固定权重的切换位置。
    # 如果 USE_UNCERTAINTY_LOSS = True，使用自动学习权重。
    # 如果 USE_UNCERTAINTY_LOSS = False，使用 DTI_LOSS_WEIGHT 和 DTA_LOSS_WEIGHT。
    if USE_UNCERTAINTY_LOSS:
        return uncertainty_criterion(loss_dti, loss_dta)
    return DTI_LOSS_WEIGHT * loss_dti + DTA_LOSS_WEIGHT * loss_dta


def modal_consistency_loss(drug_seq_feat, drug_graph_feat):
    drug_seq_feat = F.normalize(drug_seq_feat, dim=-1)
    drug_graph_feat = F.normalize(drug_graph_feat, dim=-1)
    return 1.0 - F.cosine_similarity(drug_seq_feat, drug_graph_feat, dim=-1).mean()


def supervised_contrastive_loss(features, labels, temperature=0.1):
    features = F.normalize(features, dim=-1)
    labels = labels.view(-1, 1)

    sim = torch.matmul(features, features.T) / temperature
    logits_mask = torch.ones_like(sim) - torch.eye(sim.size(0), device=sim.device)

    positive_mask = (labels == labels.T).float() * logits_mask
    exp_sim = torch.exp(sim) * logits_mask

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8))
    positive_count = positive_mask.sum(dim=1)

    valid = positive_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    loss = -(positive_mask * log_prob).sum(dim=1) / positive_count.clamp(min=1.0)
    return loss[valid].mean()


# =========================
# 指标函数
# =========================
def ci_score(y_true, y_pred, sample_size=None, sample_seed=2026):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) <= 1:
        return 0.5

    if sample_size is not None and len(y_true) > sample_size:
        rng = np.random.default_rng(sample_seed)
        idx = rng.choice(len(y_true), size=sample_size, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    a = y_true.reshape(-1, 1)
    b = y_true.reshape(1, -1)
    mask = a < b

    a_pred = y_pred.reshape(-1, 1)
    b_pred = y_pred.reshape(1, -1)
    correct = (a_pred < b_pred) & mask

    total = np.sum(mask)
    return np.sum(correct) / total if total > 0 else 0.5


def safe_pearson(y_true, y_pred):
    if len(y_true) <= 1:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    value = pearsonr(y_true, y_pred)[0]
    return 0.0 if np.isnan(value) else value


def safe_aupr(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return 0.0
    value = average_precision_score(y_true, y_pred)
    return 0.0 if np.isnan(value) else value


def get_best_metric(metrics):
    if BEST_METRIC_NAME == "val_dta_balanced":
        return (
            0.20 * metrics["ci"]
            + 0.25 * metrics["pear"]
            + 0.20 * metrics["r2"]
            - 0.25 * metrics["rmse"]
            - 0.10 * metrics["mae"]
        )
    if BEST_METRIC_NAME == "val_balanced":
        return (
            0.20 * metrics["auc"]
            + 0.15 * metrics["f1"]
            + 0.25 * metrics["ci"]
            + 0.25 * metrics["pear"]
            - 0.15 * metrics["rmse"]
        )
    if BEST_METRIC_NAME == "val_auc":
        return metrics["auc"]
    if BEST_METRIC_NAME == "val_f1":
        return metrics["f1"]
    if BEST_METRIC_NAME == "val_r2":
        return metrics["r2"]
    if BEST_METRIC_NAME == "val_ci":
        return metrics["ci"]
    if BEST_METRIC_NAME == "val_loss":
        return -metrics["loss"]
    return metrics["ci"]


def move_batch_to_device(batch, device):
    return {
        "drug_seq": batch["drug_seq"].to(device),
        "drug_mask": batch["drug_mask"].to(device),
        "protein_seq": batch["protein_seq"].to(device),
        "protein_mask": batch["protein_mask"].to(device),
        "graph": batch["graph"].to(device),
        "label_dti": batch["label_dti"].to(device),
        "label_dta": batch["label_dta"].to(device),
        "jc_code": batch["jc_code"].to(device),
    }


# =========================
# 单轮训练/验证
# 修改：使用新模型的字典输出，并显示 tqdm 进度条和 batch loss
# =========================
def run_epoch(
    loader,
    model,
    uncertainty_criterion=None,
    optimizer=None,
    desc="",
    capture_outputs=False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    if uncertainty_criterion is not None:
        uncertainty_criterion.train() if is_train else uncertainty_criterion.eval()

    total_loss = 0.0
    yt_dti, yp_dti = [], []
    yt_dta, yp_dta = [], []
    captured_ids = []
    captured_gates = []
    attention_sample = None

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        with tqdm(loader, leave=False, desc=desc, dynamic_ncols=True, mininterval=1.0) as bar:
            for raw_batch in bar:
                batch = move_batch_to_device(raw_batch, DEVICE)

                outputs = model(
                    drug_seq=batch["drug_seq"],
                    protein_seq=batch["protein_seq"],
                    drug_mask=batch["drug_mask"],
                    protein_mask=batch["protein_mask"],
                    graph=batch["graph"],
                    jc_code=batch["jc_code"],
                    return_features=True,
                )

                dti_logit = outputs["dti_logit"]
                dta_pred = outputs["dta_pred"]

                loss_dti = F.binary_cross_entropy_with_logits(
                    dti_logit,
                    batch["label_dti"],
                    pos_weight=pos_weight,
                )
                loss_dta = dta_regression_loss(dta_pred, batch["label_dta"])

                loss = compute_multitask_loss(loss_dti, loss_dta, uncertainty_criterion)

                if USE_MODAL_CONSISTENCY:
                    loss_modal = modal_consistency_loss(
                        outputs["drug_seq_feat"],
                        outputs["drug_graph_feat"],
                    )
                    loss = loss + MODAL_CONSISTENCY_WEIGHT * loss_modal

                if USE_CONTRASTIVE:
                    loss_contrast = supervised_contrastive_loss(
                        outputs["fused_feat"],
                        batch["label_dti"],
                    )
                    loss = loss + CONTRASTIVE_WEIGHT * loss_contrast

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()

                    params_for_clip = list(model.parameters())
                    if USE_UNCERTAINTY_LOSS and uncertainty_criterion is not None:
                        params_for_clip += list(uncertainty_criterion.parameters())

                    torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=5.0)
                    optimizer.step()

                batch_size = batch["label_dti"].size(0)
                total_loss += loss.item() * batch_size

                yt_dti.extend(batch["label_dti"].detach().cpu().numpy().tolist())
                yp_dti.extend(torch.sigmoid(dti_logit).detach().cpu().numpy().tolist())
                yt_dta.extend(batch["label_dta"].detach().cpu().numpy().tolist())
                yp_dta.extend(dta_pred.detach().cpu().numpy().tolist())

                if capture_outputs:
                    captured_ids.extend(raw_batch["id"])
                    captured_gates.append(outputs["modality_gates"].detach().cpu().numpy())
                    if attention_sample is None and outputs["attn_d2p"] is not None:
                        attention_sample = outputs["attn_d2p"][0].detach().cpu().numpy()

                # 修改：终端进度条中实时显示当前 batch loss
                #bar.set_postfix(loss=f"{loss.item():.4f}", dti=f"{loss_dti.item():.4f}", dta=f"{loss_dta.item():.4f}")

    yt_dti = np.asarray(yt_dti)
    yp_dti = np.asarray(yp_dti)
    yt_dta = np.asarray(yt_dta)
    yp_dta = np.asarray(yp_dta)

    yp_bin = (yp_dti > 0.5).astype(int)

    acc = accuracy_score(yt_dti, yp_bin)
    try:
        auc = roc_auc_score(yt_dti, yp_dti)
    except ValueError:
        auc = 0.5
    aupr = safe_aupr(yt_dti, yp_dti)

    pre = precision_score(yt_dti, yp_bin, zero_division=0)
    rec = recall_score(yt_dti, yp_bin, zero_division=0)
    f1 = f1_score(yt_dti, yp_bin, zero_division=0)

    mse = mean_squared_error(yt_dta, yp_dta)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yt_dta, yp_dta)
    r2 = r2_score(yt_dta, yp_dta)
    ci = ci_score(
        yt_dta,
        yp_dta,
        sample_size=TRAIN_CI_SAMPLE_SIZE if is_train else None,
    )
    pear = safe_pearson(yt_dta, yp_dta)

    avg_loss = total_loss / len(loader.dataset)

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "auc": auc,
        "aupr": aupr,
        "pre": pre,
        "rec": rec,
        "f1": f1,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "ci": ci,
        "pear": pear,
    }
    if not capture_outputs:
        return metrics
    captured = {
        "id": np.asarray(captured_ids),
        "y_true_dti": yt_dti,
        "y_pred_dti": yp_dti,
        "y_true_dta": yt_dta,
        "y_pred_dta": yp_dta,
        "modality_gates": np.concatenate(captured_gates),
        "attention_sample": attention_sample,
    }
    return metrics, captured


def compute_metrics_from_predictions(yt_dti, yp_dti, yt_dta, yp_dta, threshold=0.5):
    yp_bin = (yp_dti > threshold).astype(int)

    acc = accuracy_score(yt_dti, yp_bin)
    try:
        auc = roc_auc_score(yt_dti, yp_dti)
    except ValueError:
        auc = 0.5
    aupr = safe_aupr(yt_dti, yp_dti)

    pre = precision_score(yt_dti, yp_bin, zero_division=0)
    rec = recall_score(yt_dti, yp_bin, zero_division=0)
    f1 = f1_score(yt_dti, yp_bin, zero_division=0)

    mse = mean_squared_error(yt_dta, yp_dta)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yt_dta, yp_dta)
    r2 = r2_score(yt_dta, yp_dta)
    ci = ci_score(yt_dta, yp_dta)
    pear = safe_pearson(yt_dta, yp_dta)

    return {
        "threshold": threshold,
        "acc": acc,
        "auc": auc,
        "aupr": aupr,
        "pre": pre,
        "rec": rec,
        "f1": f1,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "ci": ci,
        "pear": pear,
    }


def collect_predictions(loader, model, desc="Predict"):
    model.eval()
    yt_dti, yp_dti = [], []
    yt_dta, yp_dta = [], []

    with torch.no_grad():
        with tqdm(loader, leave=False, desc=desc, dynamic_ncols=True, mininterval=1.0) as bar:
            for raw_batch in bar:
                batch = move_batch_to_device(raw_batch, DEVICE)
                outputs = model(
                    drug_seq=batch["drug_seq"],
                    protein_seq=batch["protein_seq"],
                    drug_mask=batch["drug_mask"],
                    protein_mask=batch["protein_mask"],
                    graph=batch["graph"],
                    jc_code=batch["jc_code"],
                    return_features=False,
                )

                yt_dti.extend(batch["label_dti"].detach().cpu().numpy().tolist())
                yp_dti.extend(torch.sigmoid(outputs["dti_logit"]).detach().cpu().numpy().tolist())
                yt_dta.extend(batch["label_dta"].detach().cpu().numpy().tolist())
                yp_dta.extend(outputs["dta_pred"].detach().cpu().numpy().tolist())

    return (
        np.asarray(yt_dti),
        np.asarray(yp_dti),
        np.asarray(yt_dta),
        np.asarray(yp_dta),
    )


def find_best_f1_threshold(yt_dti, yp_dti):
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 181):
        yp_bin = (yp_dti > threshold).astype(int)
        f1 = f1_score(yt_dti, yp_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


# =========================
# 模型与优化器
# 修改：使用新的融合模型，而不是在 train_final.py 中定义模型
# =========================
model = HarmoGATE(
    drug_vocab_size=len(DRUG_DICT) + 1,
    protein_vocab_size=len(PROT_DICT) + 1,
    graph_in_dim=26,
    embed_dim=128,
    hidden_dim=256,
    num_heads=4,
    dropout=0.2,
    use_jczs=USE_JCZS,
    use_graph=USE_GRAPH_BRANCH,
    use_cross_attention=USE_CROSS_ATTENTION,
    use_gated_fusion=USE_GATED_FUSION,
    use_transformer=USE_SEQUENCE_TRANSFORMER,
    fusion_mode=FUSION_MODE,
    use_interaction_branch=USE_INTERACTION_BRANCH,
    use_fusion_residual=USE_FUSION_RESIDUAL,
).to(DEVICE)

uncertainty_criterion = UncertaintyMultiTaskLoss().to(DEVICE) if USE_UNCERTAINTY_LOSS else None

optimizer_params = list(model.parameters())
if USE_UNCERTAINTY_LOSS and uncertainty_criterion is not None:
    optimizer_params += list(uncertainty_criterion.parameters())

optimizer = torch.optim.Adam(
    optimizer_params,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

if ARGS.preflight:
    model.train()
    raw_batch = next(iter(train_loader))
    batch = move_batch_to_device(raw_batch, DEVICE)
    outputs = model(
        drug_seq=batch["drug_seq"],
        protein_seq=batch["protein_seq"],
        drug_mask=batch["drug_mask"],
        protein_mask=batch["protein_mask"],
        graph=batch["graph"],
        jc_code=batch["jc_code"],
        return_features=True,
    )
    loss_dti = F.binary_cross_entropy_with_logits(
        outputs["dti_logit"], batch["label_dti"], pos_weight=pos_weight
    )
    loss_dta = dta_regression_loss(outputs["dta_pred"], batch["label_dta"])
    smoke_loss = compute_multitask_loss(loss_dti, loss_dta, uncertainty_criterion)
    if USE_MODAL_CONSISTENCY:
        smoke_loss = smoke_loss + MODAL_CONSISTENCY_WEIGHT * modal_consistency_loss(
            outputs["drug_seq_feat"], outputs["drug_graph_feat"]
        )
    if not torch.isfinite(smoke_loss):
        raise RuntimeError("Preflight loss is non-finite")
    optimizer.zero_grad()
    smoke_loss.backward()
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({
        "preflight": "passed",
        "dataset": DATASET,
        "split": SPLIT,
        "seed": SEED,
        "experiment": ABLATION_NAME,
        "loss": float(smoke_loss.detach().cpu()),
        "batch_size": int(batch["label_dta"].shape[0]),
        "gate_shape": list(outputs["modality_gates"].shape),
        "trainable_parameters": int(trainable_parameters),
    }, indent=2))
    sys.exit(0)


# =========================
# 日志列
# =========================
cols = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "train_auc",
    "val_auc",
    "train_aupr",
    "val_aupr",
    "train_pre",
    "val_pre",
    "train_rec",
    "val_rec",
    "train_f1",
    "val_f1",
    "train_mse",
    "val_mse",
    "train_rmse",
    "val_rmse",
    "train_mae",
    "val_mae",
    "train_r2",
    "val_r2",
    "train_ci",
    "val_ci",
    "train_pear",
    "val_pear",
    "loss_log_var_dti",
    "loss_log_var_dta",
    "total_epochs",
    "epoch_time_seconds",
]


def save_checkpoint(epoch, best_metric, finished=False):
    # 修改：完整保存续跑所需状态，避免只恢复模型导致训练状态不一致
    torch.save(model.state_dict(), latest_model_file)

    ckpt = {
        "epoch": epoch,
        "seed": SEED,
        "split_seed": ARGS.split_seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "run_id": run_id,
        "finished": finished,
        "use_uncertainty_loss": USE_UNCERTAINTY_LOSS,
        "best_metric_name": BEST_METRIC_NAME,
        "dataset": DATASET,
        "split": SPLIT,
        "data_root": DATA_ROOT,
        "dta_threshold": DTA_THRESHOLD,
        "use_jczs": USE_JCZS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "ablation_name": ABLATION_NAME,
        "use_graph_branch": USE_GRAPH_BRANCH,
        "use_cross_attention": USE_CROSS_ATTENTION,
        "use_gated_fusion": USE_GATED_FUSION,
        "use_sequence_transformer": USE_SEQUENCE_TRANSFORMER,
        "use_interaction_branch": USE_INTERACTION_BRANCH,
        "use_fusion_residual": USE_FUSION_RESIDUAL,
        "fusion_mode": FUSION_MODE,
        "task_mode": TASK_MODE,
        "use_modal_consistency": USE_MODAL_CONSISTENCY,
        "modal_consistency_weight": MODAL_CONSISTENCY_WEIGHT,
        "dti_loss_weight": DTI_LOSS_WEIGHT,
        "dta_loss_weight": DTA_LOSS_WEIGHT,
        "dta_label_weighting": ARGS.dta_label_weighting,
        # 新增：保存早停计数器，保证中断续跑后早停状态不丢失
        "early_stop_counter": early_stop_counter,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }

    if USE_UNCERTAINTY_LOSS and uncertainty_criterion is not None:
        ckpt["criterion_state_dict"] = uncertainty_criterion.state_dict()

    torch.save(ckpt, latest_ckpt_file)


# =========================
# 续跑加载
# 修改：如果上一次训练未完成，则恢复 model、optimizer、criterion、epoch、日志
# =========================
start_epoch = 1
log_rows = []
best_metric = -float("inf")
best_epoch = 0
early_stop_counter = 0
training_already_finished = False

if resume:
    checkpoint = torch.load(latest_ckpt_file, map_location=DEVICE, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if USE_UNCERTAINTY_LOSS and uncertainty_criterion is not None and "criterion_state_dict" in checkpoint:
        uncertainty_criterion.load_state_dict(checkpoint["criterion_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    training_already_finished = bool(checkpoint.get("finished", False))
    if training_already_finished:
        start_epoch = EPOCHS + 1
    best_metric = float(checkpoint.get("best_metric", -float("inf")))
    best_epoch = int(checkpoint.get("best_epoch", 0))

    # 新增：恢复早停计数器
    early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_random_state_all"]]
        )

    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        for col in cols:
            if col not in log_df.columns:
                log_df[col] = np.nan
        log_rows = log_df[cols].values.tolist()

    if training_already_finished:
        print(f"Resume run {run_id} at the pending final-test stage.")
    else:
        print(f"Resume run {run_id} from epoch {start_epoch}.")
else:
    print(f"Start new run {run_id}.")


if EVAL_ONLY_RUN_ID is not None:
    print(f"\n===== Eval Only Run {run_id} =====")
    model.load_state_dict(torch.load(best_model_file, map_location=DEVICE))

    val_yt_dti, val_yp_dti, val_yt_dta, val_yp_dta = collect_predictions(
        val_loader,
        model,
        desc=f"Val predict run {run_id}",
    )
    best_threshold, best_val_f1 = find_best_f1_threshold(val_yt_dti, val_yp_dti)
    val_metrics_tuned = compute_metrics_from_predictions(
        val_yt_dti,
        val_yp_dti,
        val_yt_dta,
        val_yp_dta,
        threshold=best_threshold,
    )

    test_yt_dti, test_yp_dti, test_yt_dta, test_yp_dta = collect_predictions(
        test_loader,
        model,
        desc=f"Test predict run {run_id}",
    )
    test_metrics_default = compute_metrics_from_predictions(
        test_yt_dti,
        test_yp_dti,
        test_yt_dta,
        test_yp_dta,
        threshold=0.5,
    )
    test_metrics_tuned = compute_metrics_from_predictions(
        test_yt_dti,
        test_yp_dti,
        test_yt_dta,
        test_yp_dta,
        threshold=best_threshold,
    )

    print(f"Best validation F1 threshold: {best_threshold:.3f} | Val F1={best_val_f1:.4f}")
    print(
        f"Val tuned DTI: AUC={val_metrics_tuned['auc']:.4f}, "
        f"Acc={val_metrics_tuned['acc']:.4f}, F1={val_metrics_tuned['f1']:.4f}, "
        f"Precision={val_metrics_tuned['pre']:.4f}, Recall={val_metrics_tuned['rec']:.4f}"
    )
    print(
        f"Test default threshold 0.500: "
        f"AUC={test_metrics_default['auc']:.4f}, Acc={test_metrics_default['acc']:.4f}, "
        f"F1={test_metrics_default['f1']:.4f}, Precision={test_metrics_default['pre']:.4f}, "
        f"Recall={test_metrics_default['rec']:.4f}"
    )
    print(
        f"Test tuned threshold {best_threshold:.3f}: "
        f"AUC={test_metrics_tuned['auc']:.4f}, Acc={test_metrics_tuned['acc']:.4f}, "
        f"F1={test_metrics_tuned['f1']:.4f}, Precision={test_metrics_tuned['pre']:.4f}, "
        f"Recall={test_metrics_tuned['rec']:.4f}"
    )
    print(
        f"Test DTA: RMSE={test_metrics_tuned['rmse']:.4f}, MAE={test_metrics_tuned['mae']:.4f}, "
        f"R2={test_metrics_tuned['r2']:.4f}, CI={test_metrics_tuned['ci']:.4f}, "
        f"Pearson={test_metrics_tuned['pear']:.4f}"
    )

    pd.DataFrame(
        [
            {"split": "val", "threshold_type": "best_val_f1", **val_metrics_tuned},
            {"split": "test", "threshold_type": "default_0.5", **test_metrics_default},
            {"split": "test", "threshold_type": "best_val_f1", **test_metrics_tuned},
        ]
    ).to_csv(f"output/eval_threshold_run{run_id}.csv", index=False)
    print(f"Saved threshold evaluation to output/eval_threshold_run{run_id}.csv")
    sys.exit(0)



# =========================
# 训练循环
# =========================


for epoch in range(start_epoch, EPOCHS + 1):
    epoch_started = time.perf_counter()
    train_metrics = run_epoch(train_loader, model, uncertainty_criterion, optimizer, desc=f"Train {epoch}/{EPOCHS}")
    val_metrics = run_epoch(val_loader, model, uncertainty_criterion, optimizer=None, desc=f"Val {epoch}/{EPOCHS}")
    epoch_time_seconds = time.perf_counter() - epoch_started

    current_metric = get_best_metric(val_metrics)

    if current_metric > best_metric:
        best_metric = current_metric
        best_epoch = epoch
        early_stop_counter = 0
        torch.save(model.state_dict(), best_model_file)
    else:
        early_stop_counter += 1

    save_checkpoint(epoch, best_metric, finished=False)

    if USE_UNCERTAINTY_LOSS and uncertainty_criterion is not None:
        log_var_dti = uncertainty_criterion.log_var_dti.item()
        log_var_dta = uncertainty_criterion.log_var_dta.item()
    else:
        log_var_dti = 0.0
        log_var_dta = 0.0

    row = [
        epoch,
        train_metrics["loss"],
        val_metrics["loss"],
        train_metrics["acc"],
        val_metrics["acc"],
        train_metrics["auc"],
        val_metrics["auc"],
        train_metrics["aupr"],
        val_metrics["aupr"],
        train_metrics["pre"],
        val_metrics["pre"],
        train_metrics["rec"],
        val_metrics["rec"],
        train_metrics["f1"],
        val_metrics["f1"],
        train_metrics["mse"],
        val_metrics["mse"],
        train_metrics["rmse"],
        val_metrics["rmse"],
        train_metrics["mae"],
        val_metrics["mae"],
        train_metrics["r2"],
        val_metrics["r2"],
        train_metrics["ci"],
        val_metrics["ci"],
        train_metrics["pear"],
        val_metrics["pear"],
        log_var_dti,
        log_var_dta,
        EPOCHS,
        epoch_time_seconds,
    ]

    log_rows.append(row)
    pd.DataFrame(log_rows, columns=cols).to_csv(log_file, index=False)

    # 修改：保留旧版风格的单行终端输出，但改成清晰不乱码
    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"TrainLoss {train_metrics['loss']:.4f} | ValLoss {val_metrics['loss']:.4f} | "
        f"DTI(AUC:{val_metrics['auc']:.4f}, AUPR:{val_metrics['aupr']:.4f}, "
        f"F1:{val_metrics['f1']:.4f}, Acc:{val_metrics['acc']:.4f}) | "
        f"DTA(RMSE:{val_metrics['rmse']:.4f}, MAE:{val_metrics['mae']:.4f}, "
        f"R2:{val_metrics['r2']:.4f}, CI:{val_metrics['ci']:.4f}) | "
        f"Best {BEST_METRIC_NAME}:{best_metric:.4f} | "
        f"EarlyStop {early_stop_counter}/{EARLY_STOP_PATIENCE}"
    )

    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch}. Best {BEST_METRIC_NAME}: {best_metric:.4f}")
        save_checkpoint(epoch, best_metric, finished=True)
        break
else:
    if not training_already_finished:
        save_checkpoint(EPOCHS, best_metric, finished=True)


# =========================
# 测试
# =========================
print("\n===== Test Results =====")
model.load_state_dict(torch.load(best_model_file, map_location=DEVICE))
old_return_attention = model.cross_attention.return_attention
model.cross_attention.return_attention = USE_INTERACTION_BRANCH and USE_CROSS_ATTENTION
test_started = time.perf_counter()
test_metrics, test_outputs = run_epoch(
    test_loader,
    model,
    uncertainty_criterion,
    optimizer=None,
    desc="Test",
    capture_outputs=True,
)

test_time_seconds = time.perf_counter() - test_started
model.cross_attention.return_attention = old_return_attention
expected_ids = test_df["id"].to_numpy() if "id" in test_df.columns else np.arange(len(test_df))
if not np.array_equal(test_outputs["id"].astype(str), expected_ids.astype(str)):
    raise RuntimeError("Test prediction order does not match test.csv")
for output_name in ("y_true_dti", "y_pred_dti", "y_true_dta", "y_pred_dta"):
    if not np.isfinite(test_outputs[output_name]).all():
        raise RuntimeError(f"Non-finite values detected in {output_name}")

print(
    f"DTI: AUC={test_metrics['auc']:.4f}, AUPR={test_metrics['aupr']:.4f}, "
    f"Acc={test_metrics['acc']:.4f}, F1={test_metrics['f1']:.4f}"
)
print(
    f"DTA: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, "
    f"R2={test_metrics['r2']:.4f}, CI={test_metrics['ci']:.4f}, Pearson={test_metrics['pear']:.4f}"
)

test_metrics_file = f"output/test_metrics_run{run_id}.csv"
pd.DataFrame([{
    **test_metrics,
    "dataset": DATASET,
    "split": SPLIT,
    "seed": SEED,
    "run_id": run_id,
    "ablation_name": ABLATION_NAME,
    "test_time_seconds": test_time_seconds,
}]).to_csv(
    test_metrics_file,
    index=False,
)
print(f"Saved test metrics to {test_metrics_file}")


# =========================
# 保存预测结果
# 修改：保存 DTI/DTA 真实值、预测值，以及多模态门控权重
# =========================
def save_final_results(outputs):
    print("Saving final outputs...")
    np.save(f"output/y_true_dti_run{run_id}.npy", outputs["y_true_dti"])
    np.save(f"output/y_pred_dti_run{run_id}.npy", outputs["y_pred_dti"])
    np.save(f"output/y_true_dta_run{run_id}.npy", outputs["y_true_dta"])
    np.save(f"output/y_pred_dta_run{run_id}.npy", outputs["y_pred_dta"])
    np.save(f"output/modality_gates_run{run_id}.npy", outputs["modality_gates"])
    if outputs["attention_sample"] is not None:
        np.save(f"output/attention_weight_run{run_id}.npy", outputs["attention_sample"])
    pd.DataFrame({
        "id": outputs["id"],
        "y_true_dti": outputs["y_true_dti"],
        "y_pred_dti": outputs["y_pred_dti"],
        "y_true_dta": outputs["y_true_dta"],
        "y_pred_dta": outputs["y_pred_dta"],
    }).to_csv(f"output/test_predictions_run{run_id}.csv", index=False)

    print(f"Saved final outputs for run {run_id}.")


save_final_results(test_outputs)

if REVISION_RUN_DIR is not None:
    with Path("run_manifest.json").open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    history_frame = pd.read_csv(log_file)
    run_config.update({
        "status": "complete",
        "best_epoch": best_epoch,
        "best_validation_score": best_metric,
        "epochs_completed": int(len(history_frame)),
        "training_time_seconds": float(history_frame["epoch_time_seconds"].sum()),
        "test_time_seconds": float(test_time_seconds),
        "test_evaluations": 1,
        "checkpoint": str(Path(best_model_file).resolve()),
        "test_metrics_file": str(Path(test_metrics_file).resolve()),
        "test_predictions_file": str(
            Path(f"output/test_predictions_run{run_id}.csv").resolve()
        ),
    })
    with Path("run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)


# =========================
# 绘图
# =========================
def plot_metric(rows, train_idx, val_idx, name, title):
    if len(rows) == 0:
        return

    epochs = [x[0] for x in rows]

    plt.figure()
    plt.plot(epochs, [x[train_idx] for x in rows], label="Train")
    plt.plot(epochs, [x[val_idx] for x in rows], label="Val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"logs/{name}_{run_id}.png")
    plt.close()


plot_metric(log_rows, 1, 2, "loss", "Loss")
plot_metric(log_rows, 5, 6, "dti_auc", "DTI AUC")
plot_metric(log_rows, 3, 4, "dti_acc", "DTI Accuracy")
plot_metric(log_rows, 11, 12, "dti_f1", "DTI F1")
plot_metric(log_rows, 15, 16, "dta_rmse", "DTA RMSE")
plot_metric(log_rows, 19, 20, "dta_r2", "DTA R2")
plot_metric(log_rows, 21, 22, "dta_ci", "DTA CI")

print("\nTraining completed.")

