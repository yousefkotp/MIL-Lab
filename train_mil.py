import argparse
import os
import json
import random
from datetime import datetime
from typing import Optional, Tuple
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
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
from src.datasets.mil_csv_dataset import MILCSVDataset
from src.builder import create_model, save_model


def _seed_worker(worker_id: int):
    """Ensure each DataLoader worker has deterministic RNG state."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def parse_args():
    p = argparse.ArgumentParser(description="Train MIL model (MeanMIL default) on CSV-defined dataset with precomputed features")
    p.add_argument('--csv_path', type=str, required=True, help='Single CSV with columns: filename,label[,case_id]; if case_id is present, all slides from a case are kept in the same fold')
    p.add_argument('--num_folds', type=int, default=5, help='Number of stratified folds (val ≈ 1/num_folds; train uses the rest). Only train/val splits are created.')
    p.add_argument('--split_seed', type=int, default=DEFAULT_SPLIT_SEED, help='Seed used only for data split so folds stay identical across runs')
    p.add_argument('--features_dir', type=str, required=True, help='Root directory containing per-slide feature files')
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--model', type=str, default='meanmil.base.slide_hubert.none', help='Model spec: name.config.encoder.task. Default: meanmil.base.slide_hubert.none')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--grad_accum_steps', type=int, default=1,
                   help='Number of steps to accumulate gradients before optimizer step')

    p.add_argument('--output_dir', type=str, default='outputs', help='Root output directory containing runs/ and checkpoints/')
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--monitor', type=str, default='val/f1_weighted', help='Metric to select best model')
    p.add_argument('--balanced_sampling', action='store_true', help='Enable class-balanced sampling for the training loader')
    return p.parse_args()


def split_df(df):
    """Split DataFrame into train/val/(optional)test using fixed column names.

    Expects columns: filename,label,split; split values: train/val[/test].
    """
    SPLIT_COL = 'split'
    TRAIN, VAL, TEST = 'train', 'val', 'test'

    df_train = df[df[SPLIT_COL] == TRAIN]
    df_val = df[df[SPLIT_COL] == VAL]
    df_test = df[df[SPLIT_COL] == TEST] if (df[SPLIT_COL] == TEST).any() else None
    return df_train, df_val, df_test


@torch.no_grad()
def _sanitize_key(s: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in s)


def evaluate(model, loader, device, num_classes, label_names: Optional[list] = None) -> Tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    # Track average number of embeddings (instances) per WSI
    n_slides = 0
    sum_embeddings = 0.0
    for feats, y, _ in loader:
        feats = feats.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # Count embeddings per slide (feats.shape is [1, M, D] per our collate)
        try:
            if feats.dim() >= 3:
                m = int(feats.shape[1])
            elif feats.dim() == 2:
                m = int(feats.shape[0])
            else:
                m = 1
            sum_embeddings += float(m)
            n_slides += 1
        except Exception:
            pass
        results, _ = model(feats, loss_fn=criterion, label=y)
        loss = results['loss']
        if loss is not None:
            total_loss += float(loss.item())
        all_logits.append(results['logits'].detach().cpu())
        all_labels.append(y.detach().cpu())
    if len(all_logits) == 0:
        return {}, np.array([]), np.array([])
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    metrics = {}
    # basic accuracies
    preds = logits.argmax(axis=1)
    metrics['loss'] = total_loss / max(1, len(loader))
    metrics['acc'] = accuracy_score(labels, preds)
    metrics['balanced_acc'] = balanced_accuracy_score(labels, preds)
    # Average number of embeddings per WSI in this split
    if n_slides > 0:
        metrics['avg_embeddings_per_wsi'] = float(sum_embeddings / float(n_slides))

    # per-class accuracy (recall per class)
    per_class_acc = []
    for c in range(num_classes):
        mask = labels == c
        denom = int(mask.sum())
        if denom > 0:
            pc = float((preds[mask] == labels[mask]).mean())
        else:
            pc = float('nan')
        per_class_acc.append(pc)
        cls_name = label_names[c] if label_names and c < len(label_names) else str(c)
        metrics[f'acc_class_{_sanitize_key(str(cls_name))}'] = pc

    # precision/recall/f1
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    metrics['precision_weighted'] = prec_w
    metrics['recall_weighted'] = rec_w
    metrics['f1_weighted'] = f1_w
    metrics['precision_macro'] = prec_m
    metrics['recall_macro'] = rec_m
    metrics['f1_macro'] = f1_m

    # ROC-AUC (macro and weighted)
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
            metrics['roc_auc_macro'] = auc
            metrics['roc_auc_weighted'] = auc  # same for binary
        else:
            auc_macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
            metrics['roc_auc_macro'] = auc_macro
            metrics['roc_auc_weighted'] = auc_weighted
    except Exception:
        metrics['roc_auc_macro'] = float('nan')
        metrics['roc_auc_weighted'] = float('nan')

    return metrics, logits, labels


def mil_collate(batch):
    # batch size is always 1; ensure features have a batch dimension
    feats, y, slide_id = batch[0]
    if feats.dim() == 2:
        feats = feats.unsqueeze(0)  # [1, M, D]
    # ensure label has batch dimension [1]
    if isinstance(y, torch.Tensor):
        if y.dim() == 0:
            y = y.unsqueeze(0).long()
        else:
            y = y.long()
    else:
        y = torch.tensor([y], dtype=torch.long)
    return feats, y, slide_id

def run_single_fold(args, fold_df: pd.DataFrame, fold_idx: int):
    # Start wall-clock timer for the entire fold (data loading, training, eval, saves)
    fold_t0 = time.time()
    # Build datasets
    base_ds = MILCSVDataset(csv_path=None, features_dir=args.features_dir, dataframe=fold_df)

    df = base_ds.df.copy()
    df_train, df_val, df_test = split_df(df)

    def make_loader(sub_df, shuffle: bool, balanced: bool = False, *, seed: Optional[int] = None):
        if sub_df is None or len(sub_df) == 0:
            return None, None
        ds = MILCSVDataset(csv_path=None, features_dir=args.features_dir, dataframe=sub_df)
        loader_seed = seed if seed is not None else args.seed
        loader_generator = _make_generator(loader_seed)
        if balanced:
            labels = ds.df['_y'].to_numpy()
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
                                batch_size=1,
                                sampler=sampler,
                                num_workers=args.num_workers,
                                collate_fn=mil_collate,
                                pin_memory=torch.cuda.is_available(),
                                persistent_workers=(args.num_workers > 0),
                                worker_init_fn=_seed_worker,
                                generator=loader_generator)
        else:
            loader = DataLoader(ds,
                                batch_size=1,
                                shuffle=shuffle,
                                num_workers=args.num_workers,
                                collate_fn=mil_collate,
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

    # determine number of classes and label names before building eval loaders
    num_classes = train_ds.num_classes
    label_names = [train_ds.label_map[i] for i in range(num_classes)]

    # training-set evaluation loader (no shuffle / no sampler)
    eval_train_loader = DataLoader(train_ds,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   collate_fn=mil_collate,
                                   pin_memory=torch.cuda.is_available(),
                                   persistent_workers=(args.num_workers > 0),
                                   worker_init_fn=_seed_worker,
                                   generator=_make_generator(fold_seed + 3))

    model = create_model(model_name=args.model, num_classes=num_classes, from_pretrained=False, keep_classifier=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Fold-specific experiment name and logging
    exp_base = args.exp_name or (f"{args.model.replace('.', '_')}-nc{num_classes}-" + datetime.now().strftime('%Y%m%d_%H%M%S'))
    exp_name = f"{exp_base}-fold{fold_idx}"
    log_root = os.path.join(args.output_dir, 'runs')
    os.makedirs(log_root, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_root, exp_name))

    best_metric = -np.inf
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        accum_steps = max(1, int(getattr(args, 'grad_accum_steps', 1)))
        num_steps = len(train_loader)
        for step_idx, (feats, y, _) in enumerate(train_loader, start=1):
            feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out, _ = model(feats, loss_fn=criterion, label=y)
            raw_loss = out['loss'] if out['loss'] is not None else criterion(out['logits'], y)
            running_loss += float(raw_loss.item())

            loss = raw_loss / float(accum_steps)
            loss.backward()

            if (step_idx % accum_steps == 0) or (step_idx == num_steps):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss_epoch = running_loss / max(1, len(train_loader))
        writer.add_scalar('train/loss', train_loss_epoch, epoch)
        # evaluate on training set for metrics tracking
        train_metrics_eval, _, _ = evaluate(model, eval_train_loader, device, num_classes, label_names=label_names)
        for k, v in train_metrics_eval.items():
            if k == 'loss':
                writer.add_scalar('train/loss_eval', v, epoch)
            else:
                writer.add_scalar(f'train/{k}', v, epoch)

        if val_loader is not None:
            val_metrics, _, _ = evaluate(model, val_loader, device, num_classes, label_names=label_names)
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
            test_metrics, _, _ = evaluate(model, test_loader, device, num_classes, label_names=label_names)
            for k, v in test_metrics.items():
                writer.add_scalar(f'test/{k}', v, epoch)

        print(f"[Fold {fold_idx}] Epoch {epoch:03d} | train_loss={train_loss_epoch:.4f} | best_monitor={best_metric:.4f}")

    # Save best model for this fold
    ckpt_root = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_dir = os.path.join(ckpt_root, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, os.path.join(ckpt_dir, 'best_state_dict.pt'))
    try:
        save_model(model, model_name=exp_name, save_folder=ckpt_root, save_pretrained=False)
    except Exception:
        pass

    # Evaluate the best checkpoint on train/val(/test) and print
    if best_state is not None:
        model.load_state_dict(best_state)
    monitor_key = args.monitor.split('/', 1)[-1]
    # fresh eval loaders (no sampler/shuffle)
    eval_train_loader = DataLoader(train_ds,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   collate_fn=mil_collate,
                                   pin_memory=torch.cuda.is_available(),
                                   persistent_workers=(args.num_workers > 0),
                                   worker_init_fn=_seed_worker,
                                   generator=_make_generator(fold_seed + 4))
    train_metrics, _, _ = evaluate(model, eval_train_loader, device, num_classes, label_names=label_names)
    val_metrics, _, _ = evaluate(model, val_loader, device, num_classes, label_names=label_names) if val_loader is not None else ({}, None, None)
    test_metrics, _, _ = evaluate(model, test_loader, device, num_classes, label_names=label_names) if test_loader is not None else ({}, None, None)

    # Fold timing (wall-clock) and logging
    fold_elapsed_s = max(0.0, float(time.time() - fold_t0))
    fold_elapsed_h = fold_elapsed_s / 3600.0
    # Log total wall time into TensorBoard for convenience (once, at final epoch index)
    try:
        writer.add_scalar('time/total_seconds', fold_elapsed_s, args.epochs)
        writer.add_scalar('time/total_hours', fold_elapsed_h, args.epochs)
    except Exception:
        pass
    # Include timing as a metric under 'val' so downstream comparison can pick it up
    # without modifying comparison tooling extensively.
    if isinstance(val_metrics, dict):
        val_metrics['train_time_hours'] = float(fold_elapsed_h)

    print(f"[Fold {fold_idx}] Best checkpoint metrics (monitor='{monitor_key}'):")
    if train_metrics:
        print("  Train:", {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in train_metrics.items()})
    if val_metrics:
        print("  Val:", {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in val_metrics.items()})
    if test_metrics:
        print("  Test:", {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in test_metrics.items()})
    # Also print timing summary for the fold
    print(f"  Timing: wall_time_hours={fold_elapsed_h:.4f} (≈ {fold_elapsed_s:.0f} s)")

    # Save fold metrics to JSON
    fold_summary = {
        'fold': fold_idx,
        'monitor': monitor_key,
        'best_monitor_value': float(best_metric) if isinstance(best_metric, (int, float, np.floating)) else best_metric,
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        # Timing block for robustness/clarity
        'timing': {
            'start_time': datetime.fromtimestamp(fold_t0).isoformat(timespec='seconds'),
            'end_time': datetime.fromtimestamp(fold_t0 + fold_elapsed_s).isoformat(timespec='seconds'),
            'wall_time_seconds': float(fold_elapsed_s),
            'wall_time_hours': float(fold_elapsed_h),
        },
        # Flat key for simple consumers
        'train_time_hours': float(fold_elapsed_h),
    }
    try:
        with open(os.path.join(ckpt_dir, 'best_metrics.json'), 'w') as f:
            json.dump(fold_summary, f, indent=2)
    except Exception:
        pass

    writer.close()

    return fold_summary


def main():
    args = parse_args()
    run_t0 = time.time()
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Reproducibility: deterministic cuDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
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

    all_fold_summaries = []
    for i, fold_df in enumerate(fold_dfs, start=1):
        print(f"==== Running fold {i}/{len(fold_dfs)} ====")
        fold_res = run_single_fold(args, fold_df, fold_idx=i)
        all_fold_summaries.append(fold_res)

    # Aggregate metrics across folds
    def _agg(split: str):
        # collect per-metric values across folds
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

    # Also include a cross-fold total training time metric in the 'val' aggregate for downstream tools.
    try:
        fold_times_h = [
            float(res.get('timing', {}).get('wall_time_hours'))
            for res in all_fold_summaries
            if isinstance(res, dict) and isinstance(res.get('timing'), dict) and isinstance(res.get('timing', {}).get('wall_time_hours'), (int, float))
        ]
        if fold_times_h:
            total_h = float(np.sum(fold_times_h))
            # Ensure dict exists
            if isinstance(agg_val, dict):
                agg_val = dict(agg_val)
                agg_val['train_time_hours'] = {
                    'mean': float(np.mean(fold_times_h)),
                    'std': float(np.std(fold_times_h, ddof=0)) if len(fold_times_h) > 1 else 0.0,
                    'n_folds': len(fold_times_h),
                }
                # Add a total across folds (std not applicable for a sum)
                agg_val['train_time_total_hours'] = {
                    'mean': total_h,
                    'std': None,
                    'n_folds': len(fold_times_h),
                }
    except Exception:
        pass

    # Print aggregated results
    print("==== Cross-fold summary (means ± std) ====")
    def _round_safe(x):
        return round(x, 4) if isinstance(x, (int, float, np.floating)) else None
    if agg_train:
        print("Train:", {k: {'mean': _round_safe(v.get('mean')), 'std': _round_safe(v.get('std')), 'n': v.get('n_folds')} for k, v in agg_train.items()})
    if agg_val:
        print("Val:", {k: {'mean': _round_safe(v.get('mean')), 'std': _round_safe(v.get('std')), 'n': v.get('n_folds')} for k, v in agg_val.items()})
    if agg_test:
        print("Test:", {k: {'mean': _round_safe(v.get('mean')), 'std': _round_safe(v.get('std')), 'n': v.get('n_folds')} for k, v in agg_test.items()})

    # Save summary JSON alongside checkpoints root
    run_elapsed_s = max(0.0, float(time.time() - run_t0))
    run_elapsed_h = run_elapsed_s / 3600.0
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
            'start_time': datetime.fromtimestamp(run_t0).isoformat(timespec='seconds'),
            'end_time': datetime.fromtimestamp(run_t0 + run_elapsed_s).isoformat(timespec='seconds'),
            'wall_time_seconds': float(run_elapsed_s),
            'wall_time_hours': float(run_elapsed_h),
        },
    }
    try:
        out_summary_path = os.path.join(args.output_dir, 'fold_summary.json')
        with open(out_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass


if __name__ == '__main__':
    main()
