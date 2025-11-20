import argparse
import os
import json
import random
from datetime import datetime
from typing import Optional, Tuple
import time

import numpy as np
import pandas as pd

# Ensure deterministic CuBLAS algorithms when using torch.use_deterministic_algorithms(True)
# Must be set before any CuBLAS handle is initialized (i.e., before first CUDA matmul op).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
)

from src.datasets.fold_utils import DEFAULT_SPLIT_SEED, prepare_folds
from src.datasets.linear_csv_dataset import LinearCSVDataset


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def parse_args():
    p = argparse.ArgumentParser(description="Train linear probe on per-WSI vector features (.h5 with 'features'=(D,))")
    p.add_argument('--csv_path', type=str, required=True, help='Single CSV with columns: filename,label[,case_id]; if case_id is present, all slides from a case are kept in the same fold')
    p.add_argument('--num_folds', type=int, default=5, help='Number of stratified folds (val ≈ 1/num_folds; train uses the rest). Only train/val splits are created.')
    p.add_argument('--split_seed', type=int, default=DEFAULT_SPLIT_SEED, help='Seed used only for data split so folds stay identical across runs')
    p.add_argument('--features_dir', type=str, required=True, help='Root directory containing per-slide feature files')
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--balanced_sampling', action='store_true', help='Enable class-balanced sampling for the training loader')
    p.add_argument('--normalize', action='store_true', help='Standardize features via training-set mean/std')

    p.add_argument('--output_dir', type=str, default='outputs', help='Root output directory containing runs/ and checkpoints/')
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--monitor', type=str, default='val/f1_weighted', help='Metric to select best model')
    return p.parse_args()


@torch.no_grad()
def _sanitize_key(s: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in s)


def _collate(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)  # [B, D]
    labels = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    slide_ids = [b[2] for b in batch]
    return feats, labels, slide_ids


def evaluate(model: nn.Module, loader: DataLoader, device, num_classes: int, label_names: Optional[list] = None, *, mean=None, std=None) -> Tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    n_slides = 0
    sum_embeddings = 0.0
    for feats, y, _ in loader:
        feats = feats.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # Normalize if provided
        if mean is not None and std is not None:
            feats = (feats - mean) / (std + 1e-8)
        logits = model(feats)
        loss = criterion(logits, y)
        total_loss += float(loss.item())
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
        # Linear probe uses one vector per WSI
        sum_embeddings += float(1.0) * feats.shape[0]
        n_slides += feats.shape[0]

    if len(all_logits) == 0:
        return {}, np.array([]), np.array([])
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    metrics = {}
    preds = logits.argmax(axis=1)
    metrics['loss'] = total_loss / max(1, len(loader))
    metrics['acc'] = accuracy_score(labels, preds)
    metrics['balanced_acc'] = balanced_accuracy_score(labels, preds)
    if n_slides > 0:
        metrics['avg_embeddings_per_wsi'] = float(sum_embeddings / float(n_slides))

    # per-class accuracy
    for c in range(num_classes):
        mask = labels == c
        denom = int(mask.sum())
        pc = float((preds[mask] == labels[mask]).mean()) if denom > 0 else float('nan')
        cls_name = label_names[c] if label_names and c < len(label_names) else str(c)
        metrics[f'acc_class_{_sanitize_key(str(cls_name))}'] = pc

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    metrics['precision_weighted'] = prec_w
    metrics['recall_weighted'] = rec_w
    metrics['f1_weighted'] = f1_w
    metrics['precision_macro'] = prec_m
    metrics['recall_macro'] = rec_m
    metrics['f1_macro'] = f1_m

    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
            metrics['roc_auc_macro'] = auc
            metrics['roc_auc_weighted'] = auc
        else:
            auc_macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
            metrics['roc_auc_macro'] = auc_macro
            metrics['roc_auc_weighted'] = auc_weighted
    except Exception:
        metrics['roc_auc_macro'] = float('nan')
        metrics['roc_auc_weighted'] = float('nan')

    return metrics, logits, labels


def split_df(df):
    SPLIT_COL = 'split'
    TRAIN, VAL, TEST = 'train', 'val', 'test'
    df_train = df[df[SPLIT_COL] == TRAIN]
    df_val = df[df[SPLIT_COL] == VAL]
    df_test = df[df[SPLIT_COL] == TEST] if (df[SPLIT_COL] == TEST).any() else None
    return df_train, df_val, df_test


def run_single_fold(args, fold_df: pd.DataFrame, fold_idx: int):
    fold_t0 = time.time()

    base_ds = LinearCSVDataset(csv_path=None, features_dir=args.features_dir, dataframe=fold_df)
    df = base_ds.df.copy()
    df_train, df_val, df_test = split_df(df)

    def make_loader(sub_df, shuffle: bool, balanced: bool = False, *, seed: Optional[int] = None):
        if sub_df is None or len(sub_df) == 0:
            return None, None
        ds = LinearCSVDataset(csv_path=None, features_dir=args.features_dir, dataframe=sub_df)
        labels = ds.df['_y'].to_numpy()
        loader_seed = seed if seed is not None else args.seed
        loader_generator = _make_generator(loader_seed)
        if balanced:
            class_counts = np.bincount(labels, minlength=ds.num_classes)
            class_counts = np.clip(class_counts, 1, None)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(labels),
                replacement=True,
                generator=_make_generator(loader_seed),
            )
            loader = DataLoader(ds,
                                batch_size=args.batch_size,
                                sampler=sampler,
                                num_workers=args.num_workers,
                                collate_fn=_collate,
                                pin_memory=torch.cuda.is_available(),
                                persistent_workers=(args.num_workers > 0),
                                worker_init_fn=_seed_worker,
                                generator=loader_generator)
        else:
            loader = DataLoader(ds,
                                batch_size=args.batch_size,
                                shuffle=shuffle,
                                num_workers=args.num_workers,
                                collate_fn=_collate,
                                pin_memory=torch.cuda.is_available(),
                                persistent_workers=(args.num_workers > 0),
                                worker_init_fn=_seed_worker,
                                generator=loader_generator)
        return ds, loader

    fold_seed = args.seed + 1000 * (fold_idx - 1)
    train_ds, train_loader = make_loader(df_train, shuffle=True, balanced=args.balanced_sampling, seed=fold_seed)
    if train_ds is None or train_loader is None:
        raise ValueError("Training split is empty after filtering; cannot proceed with training.")
    val_ds, val_loader = make_loader(df_val, shuffle=False, seed=fold_seed + 1)
    test_ds, test_loader = make_loader(df_test, shuffle=False, seed=fold_seed + 2) if df_test is not None else (None, None)

    num_classes = train_ds.num_classes
    label_names = [train_ds.label_map[i] for i in range(num_classes)]

    # Precompute normalization stats on training set if requested
    feat_dim = None
    with torch.no_grad():
        xs = []
        for feats, _, _ in DataLoader(train_ds, batch_size=min(1024, len(train_ds)), shuffle=False, num_workers=0, collate_fn=_collate):
            xs.append(feats)
        X = torch.cat(xs, dim=0) if xs else None
        if X is not None:
            feat_dim = int(X.shape[1])
    mean = std = None
    if args.normalize and feat_dim is not None:
        with torch.no_grad():
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, keepdim=True)

    # Simple linear classifier
    if feat_dim is None:
        # fallback: infer by a single item
        sample_x, _, _ = train_ds[0]
        feat_dim = int(sample_x.shape[0])
    model = nn.Linear(feat_dim, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    exp_base = args.exp_name or (f"linear_probe-nc{num_classes}-" + datetime.now().strftime('%Y%m%d_%H%M%S'))
    exp_name = f"{exp_base}-fold{fold_idx}"
    log_root = os.path.join(args.output_dir, 'runs')
    os.makedirs(log_root, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_root, exp_name))

    best_metric = -np.inf
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for feats, y, _ in train_loader:
            feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mean is not None and std is not None:
                feats = (feats - mean) / (std + 1e-8)
            logits = model(feats)
            loss = criterion(logits, y)
            running_loss += float(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_loss_epoch = running_loss / max(1, len(train_loader))
        writer.add_scalar('train/loss', train_loss_epoch, epoch)

        # Evaluate on train/val/test
        train_metrics_eval, _, _ = evaluate(model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=_collate), device, num_classes, label_names=label_names, mean=mean, std=std)
        for k, v in train_metrics_eval.items():
            tag = 'train/loss_eval' if k == 'loss' else f'train/{k}'
            writer.add_scalar(tag, v, epoch)

        if val_loader is not None:
            val_metrics, _, _ = evaluate(model, val_loader, device, num_classes, label_names=label_names, mean=mean, std=std)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            monitor_val = val_metrics.get(args.monitor.split('/', 1)[-1], float('nan'))
            current = monitor_val if isinstance(monitor_val, (int, float)) and not np.isnan(monitor_val) else -np.inf
        else:
            current = train_loss_epoch

        is_best = current > best_metric
        if is_best:
            best_metric = current
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if test_loader is not None:
            test_metrics, _, _ = evaluate(model, test_loader, device, num_classes, label_names=label_names, mean=mean, std=std)
            for k, v in test_metrics.items():
                writer.add_scalar(f'test/{k}', v, epoch)

        print(f"[Fold {fold_idx}] Epoch {epoch:03d} | train_loss={train_loss_epoch:.4f} | best_monitor={best_metric:.4f}")

    # Save checkpoint and evaluate the best
    ckpt_root = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_dir = os.path.join(ckpt_root, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, os.path.join(ckpt_dir, 'best_state_dict.pt'))

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=_collate)
    train_metrics, _, _ = evaluate(model, eval_train_loader, device, num_classes, label_names=label_names, mean=mean, std=std)
    val_metrics, _, _ = evaluate(model, val_loader, device, num_classes, label_names=label_names, mean=mean, std=std) if val_loader is not None else ({}, None, None)
    test_metrics, _, _ = evaluate(model, test_loader, device, num_classes, label_names=label_names, mean=mean, std=std) if test_loader is not None else ({}, None, None)

    fold_elapsed_s = max(0.0, float(time.time() - fold_t0))
    fold_elapsed_h = fold_elapsed_s / 3600.0
    try:
        writer.add_scalar('time/total_seconds', fold_elapsed_s, args.epochs)
        writer.add_scalar('time/total_hours', fold_elapsed_h, args.epochs)
    except Exception:
        pass
    if isinstance(val_metrics, dict):
        val_metrics['train_time_hours'] = float(fold_elapsed_h)

    monitor_key = args.monitor.split('/', 1)[-1]
    print(f"[Fold {fold_idx}] Best checkpoint metrics (monitor='{monitor_key}'):")
    if train_metrics:
        print("  Train:", {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in train_metrics.items()})
    if val_metrics:
        print("  Val:", {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in val_metrics.items()})
    if test_metrics:
        print("  Test:", {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in test_metrics.items()})
    print(f"  Timing: wall_time_hours={fold_elapsed_h:.4f} (≈ {fold_elapsed_s:.0f} s)")

    fold_summary = {
        'fold': fold_idx,
        'monitor': monitor_key,
        'best_monitor_value': float(best_metric) if isinstance(best_metric, (int, float, np.floating)) else best_metric,
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'timing': {
            'start_time': datetime.fromtimestamp(fold_t0).isoformat(timespec='seconds'),
            'end_time': datetime.fromtimestamp(fold_t0 + fold_elapsed_s).isoformat(timespec='seconds'),
            'wall_time_seconds': float(fold_elapsed_s),
            'wall_time_hours': float(fold_elapsed_h),
        },
        'train_time_hours': float(fold_elapsed_h),
    }

    writer.close()
    return fold_summary


def main():
    args = parse_args()
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    try:
        fold_dfs, _ = prepare_folds(
            csv_path=args.csv_path,
            num_folds=args.num_folds,
            split_seed=args.split_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not fold_dfs:
        raise ValueError("No folds resolved from provided CSV arguments")
    print(f"Prepared {len(fold_dfs)} folds (split_seed={args.split_seed}; val≈1/{args.num_folds}; grouped by case_id when present) from {args.csv_path}")

    t0 = time.time()
    all_fold_summaries = []
    for i, fold_df in enumerate(fold_dfs, start=1):
        print(f"==== Running fold {i}/{len(fold_dfs)} ====")
        fold_res = run_single_fold(args, fold_df, fold_idx=i)
        all_fold_summaries.append(fold_res)

    def _agg(split: str):
        values_by_key = {}
        for res in all_fold_summaries:
            metrics = res.get(split) or {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating)) and not (isinstance(v, float) and (np.isnan(v))):
                    values_by_key.setdefault(k, []).append(float(v))
        mean_std = {}
        for k, vals in values_by_key.items():
            if len(vals) > 0:
                mean_std[k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=0)),
                    'n_folds': len(vals),
                }
        return mean_std

    agg_train = _agg('train')
    agg_val = _agg('val')
    agg_test = _agg('test')

    # Attach timing aggregates to val
    try:
        fold_times_h = [
            float(res.get('timing', {}).get('wall_time_hours'))
            for res in all_fold_summaries
            if isinstance(res, dict) and isinstance(res.get('timing'), dict) and isinstance(res.get('timing', {}).get('wall_time_hours'), (int, float))
        ]
        if fold_times_h:
            total_h = float(np.sum(fold_times_h))
            agg_val = dict(agg_val)
            agg_val['train_time_hours'] = {
                'mean': float(np.mean(fold_times_h)),
                'std': float(np.std(fold_times_h, ddof=0)) if len(fold_times_h) > 1 else 0.0,
                'n_folds': len(fold_times_h),
            }
            agg_val['train_time_total_hours'] = {
                'mean': total_h,
                'std': None,
                'n_folds': len(fold_times_h),
            }
    except Exception:
        pass

    elapsed_s = max(0.0, float(time.time() - t0))
    elapsed_h = elapsed_s / 3600.0
    requested_folds = args.num_folds
    summary = {
        'monitor': args.monitor.split('/', 1)[-1],
        'num_folds': len(all_fold_summaries),
        'num_folds_requested': requested_folds,
        'split_seed': args.split_seed,
        'data_csv': args.csv_path,
        'folds': all_fold_summaries,
        'aggregate': {
            'train': agg_train,
            'val': agg_val,
            'test': agg_test,
        },
        'timing': {
            'start_time': datetime.fromtimestamp(t0).isoformat(timespec='seconds'),
            'end_time': datetime.fromtimestamp(t0 + elapsed_s).isoformat(timespec='seconds'),
            'wall_time_seconds': float(elapsed_s),
            'wall_time_hours': float(elapsed_h),
        },
    }

    try:
        out_summary_path = os.path.join(args.output_dir, 'fold_summary.json')
        os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)
        with open(out_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass


if __name__ == '__main__':
    main()
