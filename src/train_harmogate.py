import os
import math
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
from dataset import UniversalDTIDataset, dti_collate_fn, DRUG_DICT, PROT_DICT
from harmogate_model import HarmoGATE
SEED = int(os.environ.get('HARMOGATE_SEED', '42'))
EPOCHS = int(os.environ.get('HARMOGATE_EPOCHS', '100'))
BATCH_SIZE = int(os.environ.get('HARMOGATE_BATCH_SIZE', '16'))
LR = float(os.environ.get('HARMOGATE_LR', '5e-5'))
WEIGHT_DECAY = 1e-05
DATASET = os.environ.get('HARMOGATE_DATASET', 'davis').lower()
SPLIT = os.environ.get('HARMOGATE_SPLIT', 'random').lower()
DATA_ROOT = os.environ.get('HARMOGATE_DATA_ROOT', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', DATASET, SPLIT))
DTA_THRESHOLD = float(os.environ.get('HARMOGATE_THRESHOLD', '7.0' if DATASET == 'davis' else '12.1'))
USE_UNCERTAINTY_LOSS = False
DTI_LOSS_WEIGHT = 0.1
DTA_LOSS_WEIGHT = 1.0
DTA_SMOOTH_L1_WEIGHT = 0.7
DTA_MSE_WEIGHT = 0.3
USE_MODAL_CONSISTENCY = True
MODAL_CONSISTENCY_WEIGHT = 0.05
USE_CONTRASTIVE = False
CONTRASTIVE_WEIGHT = 0.02
ABLATION_NAME = os.environ.get('HARMOGATE_ABLATION', 'full')
USE_GRAPH_BRANCH = True
USE_CROSS_ATTENTION = True
USE_GATED_FUSION = True
USE_SEQUENCE_TRANSFORMER = True
if ABLATION_NAME == 'full':
    pass
elif ABLATION_NAME == 'without_graph':
    USE_GRAPH_BRANCH = False
    USE_MODAL_CONSISTENCY = False
elif ABLATION_NAME == 'without_cross_attention':
    USE_CROSS_ATTENTION = False
elif ABLATION_NAME == 'without_gated_fusion':
    USE_GATED_FUSION = False
elif ABLATION_NAME == 'without_modal_consistency':
    USE_MODAL_CONSISTENCY = False
elif ABLATION_NAME == 'without_multitask_dti':
    DTI_LOSS_WEIGHT = 0.0
elif ABLATION_NAME == 'without_transformer':
    USE_SEQUENCE_TRANSFORMER = False
else:
    raise ValueError(f'Unknown ABLATION_NAME: {ABLATION_NAME}')
EARLY_STOP_PATIENCE = 12
BEST_METRIC_NAME = 'val_dta_balanced'
TRAIN_CI_SAMPLE_SIZE = 3000
EVAL_ONLY_RUN_ID = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('output', exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

def get_run_ids():
    files = [f for f in os.listdir('logs') if f.startswith('train_log_') and f.endswith('.csv')]
    ids = []
    for f in files:
        try:
            ids.append(int(f.replace('train_log_', '').replace('.csv', '')))
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
    required = {'dataset': DATASET, 'data_root': DATA_ROOT, 'dta_threshold': DTA_THRESHOLD, 'use_uncertainty_loss': USE_UNCERTAINTY_LOSS, 'best_metric_name': BEST_METRIC_NAME, 'ablation_name': ABLATION_NAME, 'use_graph_branch': USE_GRAPH_BRANCH, 'use_cross_attention': USE_CROSS_ATTENTION, 'use_gated_fusion': USE_GATED_FUSION, 'use_sequence_transformer': USE_SEQUENCE_TRANSFORMER}
    for key, expected in required.items():
        if key not in ckpt:
            print(f'Skip resume: checkpoint missing {key}, run is not config-safe.')
            return False
        actual = ckpt.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-08:
                print(f'Skip resume: checkpoint {key}={actual} != current {expected}.')
                return False
        elif actual != expected:
            print(f'Skip resume: checkpoint {key}={actual} != current {expected}.')
            return False
    return True

def should_resume_run(last_id):
    if last_id <= 0:
        return False
    last_log = f'logs/train_log_{last_id}.csv'
    last_ckpt = f'checkpoints/latest_checkpoint_{last_id}.pth'
    if not os.path.exists(last_log) or not os.path.exists(last_ckpt):
        return False
    try:
        ckpt = torch.load(last_ckpt, map_location='cpu', weights_only=False)
        if ckpt.get('finished', False):
            return False
        if not is_checkpoint_compatible(ckpt):
            return False
        df = pd.read_csv(last_log)
        if len(df) == 0:
            return False
        last_epoch = int(df['epoch'].max())
        total_epochs = int(df['total_epochs'].iloc[-1]) if 'total_epochs' in df.columns else EPOCHS
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
log_file = f'logs/train_log_{run_id}.csv'
best_model_file = f'checkpoints/best_model_{run_id}.pth'
latest_model_file = f'checkpoints/latest_model_{run_id}.pth'
latest_ckpt_file = f'checkpoints/latest_checkpoint_{run_id}.pth'
print('=' * 70)
print(f'Run ID: {run_id}')
print(f'Dataset: {DATASET}')
print(f'Split: {SPLIT}')
print(f'Device: {DEVICE}')
print(f'Use uncertainty loss: {USE_UNCERTAINTY_LOSS}')
print(f'Ablation: {ABLATION_NAME}')
print(f'Ablation switches: graph={USE_GRAPH_BRANCH}, cross_attn={USE_CROSS_ATTENTION}, gated_fusion={USE_GATED_FUSION}, transformer={USE_SEQUENCE_TRANSFORMER}, modal_consistency={USE_MODAL_CONSISTENCY}, dti_loss_weight={DTI_LOSS_WEIGHT}')
print(f'Best metric: {BEST_METRIC_NAME}')
print(f'Resume: {resume}')
print('=' * 70)
train_df = pd.read_csv(f'{DATA_ROOT}/train.csv')
val_df = pd.read_csv(f'{DATA_ROOT}/val.csv')
test_df = pd.read_csv(f'{DATA_ROOT}/test.csv')

def get_label_column(df):
    for name in ['label', 'Label', 'affinity', 'y']:
        if name in df.columns:
            return name
    raise ValueError('No label column found. Expected one of: label, Label, affinity, y')
label_col = get_label_column(train_df)
train_labels = train_df[label_col].astype(float).values
train_dti_labels = (train_labels >= DTA_THRESHOLD).astype(float)
pos = train_dti_labels.sum()
neg = len(train_dti_labels) - pos
if pos > 0:
    pos_weight = torch.tensor([math.sqrt(neg / pos) * 0.7], dtype=torch.float32).to(DEVICE)
else:
    pos_weight = torch.tensor([1.0], dtype=torch.float32).to(DEVICE)
train_dataset = UniversalDTIDataset(train_df, dataset_type=DATASET, dta_threshold=DTA_THRESHOLD)
val_dataset = UniversalDTIDataset(val_df, dataset_type=DATASET, dta_threshold=DTA_THRESHOLD)
test_dataset = UniversalDTIDataset(test_df, dataset_type=DATASET, dta_threshold=DTA_THRESHOLD)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dti_collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dti_collate_fn, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dti_collate_fn, num_workers=0)

def dta_regression_loss(pred, true):
    mse_loss = (pred - true) ** 2
    smooth_l1_loss = F.smooth_l1_loss(pred, true, beta=0.5, reduction='none')
    weight = torch.ones_like(true)
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
        log_var_dti = torch.clamp(self.log_var_dti, min=-1.5, max=1.5)
        log_var_dta = torch.clamp(self.log_var_dta, min=-1.5, max=1.5)
        precision_dti = torch.exp(-log_var_dti)
        precision_dta = torch.exp(-log_var_dta)
        return precision_dti * loss_dti + log_var_dti + precision_dta * loss_dta + log_var_dta

def compute_multitask_loss(loss_dti, loss_dta, uncertainty_criterion=None):
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
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-08))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    loss = -(positive_mask * log_prob).sum(dim=1) / positive_count.clamp(min=1.0)
    return loss[valid].mean()

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
    if BEST_METRIC_NAME == 'val_dta_balanced':
        return 0.2 * metrics['ci'] + 0.25 * metrics['pear'] + 0.2 * metrics['r2'] - 0.25 * metrics['rmse'] - 0.1 * metrics['mae']
    if BEST_METRIC_NAME == 'val_balanced':
        return 0.2 * metrics['auc'] + 0.15 * metrics['f1'] + 0.25 * metrics['ci'] + 0.25 * metrics['pear'] - 0.15 * metrics['rmse']
    if BEST_METRIC_NAME == 'val_auc':
        return metrics['auc']
    if BEST_METRIC_NAME == 'val_f1':
        return metrics['f1']
    if BEST_METRIC_NAME == 'val_r2':
        return metrics['r2']
    if BEST_METRIC_NAME == 'val_ci':
        return metrics['ci']
    if BEST_METRIC_NAME == 'val_loss':
        return -metrics['loss']
    return metrics['ci']

def move_batch_to_device(batch, device):
    return {'drug_seq': batch['drug_seq'].to(device), 'drug_mask': batch['drug_mask'].to(device), 'protein_seq': batch['protein_seq'].to(device), 'protein_mask': batch['protein_mask'].to(device), 'graph': batch['graph'].to(device), 'label_dti': batch['label_dti'].to(device), 'label_dta': batch['label_dta'].to(device)}

def run_epoch(loader, model, uncertainty_criterion=None, optimizer=None, desc=''):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    if uncertainty_criterion is not None:
        uncertainty_criterion.train() if is_train else uncertainty_criterion.eval()
    total_loss = 0.0
    yt_dti, yp_dti = ([], [])
    yt_dta, yp_dta = ([], [])
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        with tqdm(loader, leave=False, desc=desc, dynamic_ncols=True, mininterval=1.0) as bar:
            for raw_batch in bar:
                batch = move_batch_to_device(raw_batch, DEVICE)
                outputs = model(drug_seq=batch['drug_seq'], protein_seq=batch['protein_seq'], drug_mask=batch['drug_mask'], protein_mask=batch['protein_mask'], graph=batch['graph'], return_features=True)
                dti_logit = outputs['dti_logit']
                dta_pred = outputs['dta_pred']
                loss_dti = F.binary_cross_entropy_with_logits(dti_logit, batch['label_dti'], pos_weight=pos_weight)
                loss_dta = dta_regression_loss(dta_pred, batch['label_dta'])
                loss = compute_multitask_loss(loss_dti, loss_dta, uncertainty_criterion)
                if USE_MODAL_CONSISTENCY:
                    loss_modal = modal_consistency_loss(outputs['drug_seq_feat'], outputs['drug_graph_feat'])
                    loss = loss + MODAL_CONSISTENCY_WEIGHT * loss_modal
                if USE_CONTRASTIVE:
                    loss_contrast = supervised_contrastive_loss(outputs['fused_feat'], batch['label_dti'])
                    loss = loss + CONTRASTIVE_WEIGHT * loss_contrast
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    params_for_clip = list(model.parameters())
                    if USE_UNCERTAINTY_LOSS and uncertainty_criterion is not None:
                        params_for_clip += list(uncertainty_criterion.parameters())
                    torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=5.0)
                    optimizer.step()
                batch_size = batch['label_dti'].size(0)
                total_loss += loss.item() * batch_size
                yt_dti.extend(batch['label_dti'].detach().cpu().numpy().tolist())
                yp_dti.extend(torch.sigmoid(dti_logit).detach().cpu().numpy().tolist())
                yt_dta.extend(batch['label_dta'].detach().cpu().numpy().tolist())
                yp_dta.extend(dta_pred.detach().cpu().numpy().tolist())
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
    ci = ci_score(yt_dta, yp_dta, sample_size=TRAIN_CI_SAMPLE_SIZE if is_train else None)
    pear = safe_pearson(yt_dta, yp_dta)
    avg_loss = total_loss / len(loader.dataset)
    return {'loss': avg_loss, 'acc': acc, 'auc': auc, 'aupr': aupr, 'pre': pre, 'rec': rec, 'f1': f1, 'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'ci': ci, 'pear': pear}

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
    return {'threshold': threshold, 'acc': acc, 'auc': auc, 'aupr': aupr, 'pre': pre, 'rec': rec, 'f1': f1, 'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'ci': ci, 'pear': pear}

def collect_predictions(loader, model, desc='Predict'):
    model.eval()
    yt_dti, yp_dti = ([], [])
    yt_dta, yp_dta = ([], [])
    with torch.no_grad():
        with tqdm(loader, leave=False, desc=desc, dynamic_ncols=True, mininterval=1.0) as bar:
            for raw_batch in bar:
                batch = move_batch_to_device(raw_batch, DEVICE)
                outputs = model(drug_seq=batch['drug_seq'], protein_seq=batch['protein_seq'], drug_mask=batch['drug_mask'], protein_mask=batch['protein_mask'], graph=batch['graph'], return_features=False)
                yt_dti.extend(batch['label_dti'].detach().cpu().numpy().tolist())
                yp_dti.extend(torch.sigmoid(outputs['dti_logit']).detach().cpu().numpy().tolist())
                yt_dta.extend(batch['label_dta'].detach().cpu().numpy().tolist())
                yp_dta.extend(outputs['dta_pred'].detach().cpu().numpy().tolist())
    return (np.asarray(yt_dti), np.asarray(yp_dti), np.asarray(yt_dta), np.asarray(yp_dta))

def find_best_f1_threshold(yt_dti, yp_dti):
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        yp_bin = (yp_dti > threshold).astype(int)
        f1 = f1_score(yt_dti, yp_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return (best_threshold, best_f1)
model = HarmoGATE(drug_vocab_size=len(DRUG_DICT) + 1, protein_vocab_size=len(PROT_DICT) + 1, graph_in_dim=26, embed_dim=128, hidden_dim=256, num_heads=4, dropout=0.2, use_graph=USE_GRAPH_BRANCH, use_cross_attention=USE_CROSS_ATTENTION, use_gated_fusion=USE_GATED_FUSION, use_transformer=USE_SEQUENCE_TRANSFORMER).to(DEVICE)
uncertainty_criterion = UncertaintyMultiTaskLoss().to(DEVICE) if USE_UNCERTAINTY_LOSS else None
optimizer_params = list(model.parameters())
if USE_UNCERTAINTY_LOSS and uncertainty_criterion is not None:
    optimizer_params += list(uncertainty_criterion.parameters())
optimizer = torch.optim.Adam(optimizer_params, lr=LR, weight_decay=WEIGHT_DECAY)
cols = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_auc', 'val_auc', 'train_aupr', 'val_aupr', 'train_pre', 'val_pre', 'train_rec', 'val_rec', 'train_f1', 'val_f1', 'train_mse', 'val_mse', 'train_rmse', 'val_rmse', 'train_mae', 'val_mae', 'train_r2', 'val_r2', 'train_ci', 'val_ci', 'train_pear', 'val_pear', 'loss_log_var_dti', 'loss_log_var_dta', 'total_epochs']

def save_checkpoint(epoch, best_metric, finished=False):
    torch.save(model.state_dict(), latest_model_file)
    ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_metric': best_metric, 'run_id': run_id, 'finished': finished, 'use_uncertainty_loss': USE_UNCERTAINTY_LOSS, 'best_metric_name': BEST_METRIC_NAME, 'dataset': DATASET, 'split': SPLIT, 'data_root': DATA_ROOT, 'dta_threshold': DTA_THRESHOLD, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'batch_size': BATCH_SIZE, 'ablation_name': ABLATION_NAME, 'use_graph_branch': USE_GRAPH_BRANCH, 'use_cross_attention': USE_CROSS_ATTENTION, 'use_gated_fusion': USE_GATED_FUSION, 'use_sequence_transformer': USE_SEQUENCE_TRANSFORMER, 'use_modal_consistency': USE_MODAL_CONSISTENCY, 'dti_loss_weight': DTI_LOSS_WEIGHT, 'early_stop_counter': early_stop_counter}
    if USE_UNCERTAINTY_LOSS and uncertainty_criterion is not None:
        ckpt['criterion_state_dict'] = uncertainty_criterion.state_dict()
    torch.save(ckpt, latest_ckpt_file)
start_epoch = 1
log_rows = []
best_metric = -float('inf')
early_stop_counter = 0
if resume:
    checkpoint = torch.load(latest_ckpt_file, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if USE_UNCERTAINTY_LOSS and uncertainty_criterion is not None and ('criterion_state_dict' in checkpoint):
        uncertainty_criterion.load_state_dict(checkpoint['criterion_state_dict'])
    start_epoch = int(checkpoint['epoch']) + 1
    best_metric = float(checkpoint.get('best_metric', -float('inf')))
    early_stop_counter = int(checkpoint.get('early_stop_counter', 0))
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        for col in cols:
            if col not in log_df.columns:
                log_df[col] = np.nan
        log_rows = log_df[cols].values.tolist()
    print(f'Resume run {run_id} from epoch {start_epoch}.')
else:
    print(f'Start new run {run_id}.')
if EVAL_ONLY_RUN_ID is not None:
    print(f'\n===== Eval Only Run {run_id} =====')
    model.load_state_dict(torch.load(best_model_file, map_location=DEVICE))
    val_yt_dti, val_yp_dti, val_yt_dta, val_yp_dta = collect_predictions(val_loader, model, desc=f'Val predict run {run_id}')
    best_threshold, best_val_f1 = find_best_f1_threshold(val_yt_dti, val_yp_dti)
    val_metrics_tuned = compute_metrics_from_predictions(val_yt_dti, val_yp_dti, val_yt_dta, val_yp_dta, threshold=best_threshold)
    test_yt_dti, test_yp_dti, test_yt_dta, test_yp_dta = collect_predictions(test_loader, model, desc=f'Test predict run {run_id}')
    test_metrics_default = compute_metrics_from_predictions(test_yt_dti, test_yp_dti, test_yt_dta, test_yp_dta, threshold=0.5)
    test_metrics_tuned = compute_metrics_from_predictions(test_yt_dti, test_yp_dti, test_yt_dta, test_yp_dta, threshold=best_threshold)
    print(f'Best validation F1 threshold: {best_threshold:.3f} | Val F1={best_val_f1:.4f}')
    print(f"Val tuned DTI: AUC={val_metrics_tuned['auc']:.4f}, Acc={val_metrics_tuned['acc']:.4f}, F1={val_metrics_tuned['f1']:.4f}, Precision={val_metrics_tuned['pre']:.4f}, Recall={val_metrics_tuned['rec']:.4f}")
    print(f"Test default threshold 0.500: AUC={test_metrics_default['auc']:.4f}, Acc={test_metrics_default['acc']:.4f}, F1={test_metrics_default['f1']:.4f}, Precision={test_metrics_default['pre']:.4f}, Recall={test_metrics_default['rec']:.4f}")
    print(f"Test tuned threshold {best_threshold:.3f}: AUC={test_metrics_tuned['auc']:.4f}, Acc={test_metrics_tuned['acc']:.4f}, F1={test_metrics_tuned['f1']:.4f}, Precision={test_metrics_tuned['pre']:.4f}, Recall={test_metrics_tuned['rec']:.4f}")
    print(f"Test DTA: RMSE={test_metrics_tuned['rmse']:.4f}, MAE={test_metrics_tuned['mae']:.4f}, R2={test_metrics_tuned['r2']:.4f}, CI={test_metrics_tuned['ci']:.4f}, Pearson={test_metrics_tuned['pear']:.4f}")
    pd.DataFrame([{'split': 'val', 'threshold_type': 'best_val_f1', **val_metrics_tuned}, {'split': 'test', 'threshold_type': 'default_0.5', **test_metrics_default}, {'split': 'test', 'threshold_type': 'best_val_f1', **test_metrics_tuned}]).to_csv(f'output/eval_threshold_run{run_id}.csv', index=False)
    print(f'Saved threshold evaluation to output/eval_threshold_run{run_id}.csv')
    sys.exit(0)
for epoch in range(start_epoch, EPOCHS + 1):
    train_metrics = run_epoch(train_loader, model, uncertainty_criterion, optimizer, desc=f'Train {epoch}/{EPOCHS}')
    val_metrics = run_epoch(val_loader, model, uncertainty_criterion, optimizer=None, desc=f'Val {epoch}/{EPOCHS}')
    current_metric = get_best_metric(val_metrics)
    if current_metric > best_metric:
        best_metric = current_metric
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
    row = [epoch, train_metrics['loss'], val_metrics['loss'], train_metrics['acc'], val_metrics['acc'], train_metrics['auc'], val_metrics['auc'], train_metrics['aupr'], val_metrics['aupr'], train_metrics['pre'], val_metrics['pre'], train_metrics['rec'], val_metrics['rec'], train_metrics['f1'], val_metrics['f1'], train_metrics['mse'], val_metrics['mse'], train_metrics['rmse'], val_metrics['rmse'], train_metrics['mae'], val_metrics['mae'], train_metrics['r2'], val_metrics['r2'], train_metrics['ci'], val_metrics['ci'], train_metrics['pear'], val_metrics['pear'], log_var_dti, log_var_dta, EPOCHS]
    log_rows.append(row)
    pd.DataFrame(log_rows, columns=cols).to_csv(log_file, index=False)
    print(f"Epoch {epoch:03d}/{EPOCHS} | TrainLoss {train_metrics['loss']:.4f} | ValLoss {val_metrics['loss']:.4f} | DTI(AUC:{val_metrics['auc']:.4f}, AUPR:{val_metrics['aupr']:.4f}, F1:{val_metrics['f1']:.4f}, Acc:{val_metrics['acc']:.4f}) | DTA(RMSE:{val_metrics['rmse']:.4f}, MAE:{val_metrics['mae']:.4f}, R2:{val_metrics['r2']:.4f}, CI:{val_metrics['ci']:.4f}) | Best {BEST_METRIC_NAME}:{best_metric:.4f} | EarlyStop {early_stop_counter}/{EARLY_STOP_PATIENCE}")
    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f'Early stopping at epoch {epoch}. Best {BEST_METRIC_NAME}: {best_metric:.4f}')
        save_checkpoint(epoch, best_metric, finished=True)
        break
else:
    save_checkpoint(EPOCHS, best_metric, finished=True)
print('\n===== Test Results =====')
model.load_state_dict(torch.load(best_model_file, map_location=DEVICE))
test_metrics = run_epoch(test_loader, model, uncertainty_criterion, optimizer=None, desc='Test')
print(f"DTI: AUC={test_metrics['auc']:.4f}, AUPR={test_metrics['aupr']:.4f}, Acc={test_metrics['acc']:.4f}, F1={test_metrics['f1']:.4f}")
print(f"DTA: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.4f}, CI={test_metrics['ci']:.4f}, Pearson={test_metrics['pear']:.4f}")
test_metrics_file = f'output/test_metrics_run{run_id}.csv'
pd.DataFrame([{**test_metrics, 'dataset': DATASET, 'split': SPLIT, 'run_id': run_id, 'ablation_name': ABLATION_NAME}]).to_csv(test_metrics_file, index=False)
print(f'Saved test metrics to {test_metrics_file}')

def save_final_results():
    model.eval()
    yt_dti_all = []
    yp_dti_all = []
    yt_dta_all = []
    yp_dta_all = []
    gates_all = []
    attention_sample = []
    old_return_attention = model.cross_attention.return_attention
    model.cross_attention.return_attention = True
    print('Saving final outputs...')
    with torch.no_grad():
        for raw_batch in test_loader:
            batch = move_batch_to_device(raw_batch, DEVICE)
            outputs = model(drug_seq=batch['drug_seq'], protein_seq=batch['protein_seq'], drug_mask=batch['drug_mask'], protein_mask=batch['protein_mask'], graph=batch['graph'], return_features=False)
            yt_dti_all.append(batch['label_dti'].cpu().numpy())
            yp_dti_all.append(torch.sigmoid(outputs['dti_logit']).cpu().numpy())
            yt_dta_all.append(batch['label_dta'].cpu().numpy())
            yp_dta_all.append(outputs['dta_pred'].cpu().numpy())
            gates_all.append(outputs['modality_gates'].cpu().numpy())
            if len(attention_sample) == 0 and outputs['attn_d2p'] is not None:
                attention_sample.append(outputs['attn_d2p'][0].detach().cpu().numpy())
    model.cross_attention.return_attention = old_return_attention
    np.save(f'output/y_true_dti_run{run_id}.npy', np.concatenate(yt_dti_all))
    np.save(f'output/y_pred_dti_run{run_id}.npy', np.concatenate(yp_dti_all))
    np.save(f'output/y_true_dta_run{run_id}.npy', np.concatenate(yt_dta_all))
    np.save(f'output/y_pred_dta_run{run_id}.npy', np.concatenate(yp_dta_all))
    np.save(f'output/modality_gates_run{run_id}.npy', np.concatenate(gates_all))
    if len(attention_sample) > 0:
        np.save(f'output/attention_weight_run{run_id}.npy', attention_sample[0])
    print(f'Saved final outputs for run {run_id}.')
save_final_results()


print('\nTraining completed.')
