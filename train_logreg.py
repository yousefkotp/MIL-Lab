import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter

from src.datasets.fold_utils import DEFAULT_SPLIT_SEED, prepare_folds
from src.datasets.logreg_csv_dataset import LogRegCSVDataset
from src.metrics_utils import compute_additional_metrics, load_config_metrics


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _sanitize_key(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def parse_args():
    p = argparse.ArgumentParser(
        description="Linear probing with multinomial logistic regression (LBFGS) and L2 sweep selected by validation loss."
    )
    p.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Single CSV with columns: filename,label[,case_id]; if case_id is present, all slides from a case are kept in the same fold",
    )
    p.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to task config.yaml (defaults to <csv_dir>/config.yaml). Must define sample_col.",
    )
    p.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of stratified folds (val ≈ 1/num_folds; train uses the rest). Only train/val splits are created.",
    )
    p.add_argument(
        "--split_seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed used only for data split so folds stay identical across runs",
    )
    p.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Root directory containing per-slide feature files",
    )
    p.add_argument(
        "--feature_parent_dir",
        type=str,
        required=True,
        help="Name of the parent directory that contains the .h5 feature files when searching recursively (e.g., 'features_lunit-vits8')",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--case_fusion",
        type=str,
        default="late",
        choices=["late", "early"],
        help="How to fuse multiple slides per case when 'case_id'/'slide_ids' are present. 'late' (default) averages per-slide embeddings.",
    )
    p.add_argument(
        "--normalize",
        "--standardize",
        dest="normalize",
        action="store_true",
        help="Apply per-sample L2 normalization to fused embeddings",
    )
    p.add_argument(
        "--no-normalize",
        "--no-standardize",
        dest="normalize",
        action="store_false",
        help="Disable L2 normalization of fused embeddings",
    )
    p.set_defaults(normalize=True)
    p.add_argument(
        "--class_weight_balanced",
        dest="class_weight_balanced",
        action="store_true",
        help="Use class_weight='balanced' for logistic regression",
    )
    p.add_argument(
        "--no-class_weight_balanced",
        dest="class_weight_balanced",
        action="store_false",
        help="Disable class_weight='balanced'",
    )
    p.set_defaults(class_weight_balanced=True)
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of CPU workers (passed to scikit-learn where applicable)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Root output directory containing runs/ and checkpoints/",
    )
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument(
        "--monitor",
        type=str,
        default="val/loss",
        help="Metric to select best model (default: val/loss; 'loss' is minimized, others maximized)",
    )
    return p.parse_args()


def _split_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    SPLIT_COL = "split"
    TRAIN, VAL, TEST = "train", "val", "test"
    df_train = df[df[SPLIT_COL] == TRAIN]
    df_val = df[df[SPLIT_COL] == VAL]
    df_test = df[df[SPLIT_COL] == TEST] if (df[SPLIT_COL] == TEST).any() else None
    return df_train, df_val, df_test


def _extract_features(ds: LogRegCSVDataset):
    feats: List[torch.Tensor] = []
    labels: List[int] = []
    for i in range(len(ds)):
        f, y, _ = ds[i]
        feats.append(f.float())
        labels.append(int(y))
    if len(feats) == 0:
        return torch.empty((0, 0)), np.array([], dtype=int)
    feats_tensor = torch.stack(feats, dim=0)
    labels_np = np.asarray(labels, dtype=int)
    return feats_tensor, labels_np


def _evaluate(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    label_names: Optional[list] = None,
    *,
    extra_metrics: Optional[list] = None,
    avg_embeddings: Optional[float] = None,
):
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        return {}, np.array([]), np.array([])

    preds = model.predict(X)
    probs = None
    try:
        probs = model.predict_proba(X)
    except Exception:
        probs = None

    metrics = {}
    try:
        if probs is not None:
            metrics["loss"] = log_loss(y, probs, labels=model.classes_)
        else:
            metrics["loss"] = float("nan")
    except Exception:
        metrics["loss"] = float("nan")

    metrics["acc"] = accuracy_score(y, preds)
    metrics["balanced_acc"] = balanced_accuracy_score(y, preds)
    if avg_embeddings is not None:
        metrics["avg_embeddings_per_wsi"] = float(avg_embeddings)

    for c in range(num_classes):
        mask = y == c
        denom = int(mask.sum())
        pc = float((preds[mask] == y[mask]).mean()) if denom > 0 else float("nan")
        cls_name = label_names[c] if label_names and c < len(label_names) else str(c)
        metrics[f"acc_class_{_sanitize_key(str(cls_name))}"] = pc

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y, preds, average="weighted", zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y, preds, average="macro", zero_division=0
    )
    metrics["precision_weighted"] = prec_w
    metrics["recall_weighted"] = rec_w
    metrics["f1_weighted"] = f1_w
    metrics["precision_macro"] = prec_m
    metrics["recall_macro"] = rec_m
    metrics["f1_macro"] = f1_m

    try:
        if probs is not None:
            if num_classes == 2:
                auc = roc_auc_score(y, probs[:, 1])
                metrics["roc_auc_macro"] = auc
                metrics["roc_auc_weighted"] = auc
            else:
                auc_macro = roc_auc_score(y, probs, multi_class="ovr", average="macro")
                auc_weighted = roc_auc_score(y, probs, multi_class="ovr", average="weighted")
                metrics["roc_auc_macro"] = auc_macro
                metrics["roc_auc_weighted"] = auc_weighted
        else:
            metrics["roc_auc_macro"] = float("nan")
            metrics["roc_auc_weighted"] = float("nan")
    except Exception:
        metrics["roc_auc_macro"] = float("nan")
        metrics["roc_auc_weighted"] = float("nan")

    if extra_metrics:
        try:
            metrics.update(compute_additional_metrics(extra_metrics, y, preds))
        except Exception:
            pass

    return metrics, preds, y


def run_single_fold(args, fold_df: pd.DataFrame, fold_idx: int):
    fold_t0 = time.time()
    extra_metrics = list(getattr(args, "custom_metrics", []) or [])

    base_ds = LogRegCSVDataset(
        csv_path=None,
        features_dir=args.features_dir,
        dataframe=fold_df,
        case_fusion=args.case_fusion,
        sample_col=args.sample_col,
        feature_parent_dir=args.feature_parent_dir,
    )
    df = base_ds.df.copy()
    df_train, df_val, df_test = _split_df(df)

    train_ds = LogRegCSVDataset(
        csv_path=None,
        features_dir=args.features_dir,
        dataframe=df_train,
        case_fusion=args.case_fusion,
        sample_col=args.sample_col,
        feature_parent_dir=args.feature_parent_dir,
    )
    if len(train_ds) == 0:
        raise ValueError("Training split is empty after filtering; cannot proceed with training.")
    val_ds = (
        LogRegCSVDataset(
            csv_path=None,
            features_dir=args.features_dir,
            dataframe=df_val,
            case_fusion=args.case_fusion,
            sample_col=args.sample_col,
            feature_parent_dir=args.feature_parent_dir,
        )
        if df_val is not None and len(df_val) > 0
        else None
    )
    test_ds = (
        LogRegCSVDataset(
            csv_path=None,
            features_dir=args.features_dir,
            dataframe=df_test,
            case_fusion=args.case_fusion,
            sample_col=args.sample_col,
            feature_parent_dir=args.feature_parent_dir,
        )
        if df_test is not None and len(df_test) > 0
        else None
    )

    num_classes = train_ds.num_classes
    label_names = [train_ds.label_map[i] for i in range(num_classes)]

    train_feats_t, train_labels = _extract_features(train_ds)
    val_feats_t, val_labels = _extract_features(val_ds) if val_ds is not None else (None, None)
    test_feats_t, test_labels = _extract_features(test_ds) if test_ds is not None else (None, None)

    if train_feats_t.numel() == 0:
        raise ValueError("No training features extracted; aborting.")

    def _norm(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if t is None:
            return None
        if t.numel() == 0:
            return np.zeros((0, t.shape[-1]), dtype=np.float32)
        if args.normalize:
            t = torch.nn.functional.normalize(t, p=2, dim=1, eps=1e-8)
        return t.cpu().numpy()

    train_feats = _norm(train_feats_t)
    val_feats = _norm(val_feats_t)
    test_feats = _norm(test_feats_t)

    avg_embeddings = {
        "train": float(train_ds.df["num_slides"].mean()) if "num_slides" in train_ds.df.columns else None
    }
    if val_ds is not None:
        avg_embeddings["val"] = float(val_ds.df["num_slides"].mean()) if "num_slides" in val_ds.df.columns else None
    if test_ds is not None:
        avg_embeddings["test"] = float(test_ds.df["num_slides"].mean()) if "num_slides" in test_ds.df.columns else None

    exp_base = args.exp_name or (f"logreg-nc{num_classes}-" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    exp_name = f"{exp_base}-fold{fold_idx}"
    log_root = os.path.join(args.output_dir, "runs")
    os.makedirs(log_root, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_root, exp_name))

    l2_grid = np.logspace(-6, 5, num=45)
    monitor_key = args.monitor.split("/", 1)[-1]
    minimize = monitor_key in ("loss", "log_loss")
    best_metric = np.inf if minimize else -np.inf
    best_model = None
    best_l2 = None
    best_val_metrics = {}
    val_available = val_feats is not None and val_labels is not None and len(val_labels) > 0

    for idx, l2_val in enumerate(l2_grid, start=1):
        C_val = 1.0 / float(l2_val)
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            multi_class="auto",
            C=C_val,
            n_jobs=args.num_workers,
            class_weight="balanced" if args.class_weight_balanced else None,
            random_state=args.seed,
        )
        model.fit(train_feats, train_labels)

        target_X = val_feats if val_available else train_feats
        target_y = val_labels if val_available else train_labels
        monitor_metrics, _, _ = _evaluate(
            model,
            target_X,
            target_y,
            num_classes,
            label_names=label_names,
            extra_metrics=extra_metrics,
            avg_embeddings=avg_embeddings.get("val") if val_available else avg_embeddings.get("train"),
        )
        current = monitor_metrics.get(monitor_key, np.inf if minimize else -np.inf)
        if not isinstance(current, (int, float, np.floating)) or np.isnan(current):
            current = np.inf if minimize else -np.inf

        writer.add_scalar(f"val/{monitor_key}", current, idx)
        writer.add_scalar("train/size", len(train_labels), idx)

        is_better = current < best_metric if minimize else current > best_metric
        if is_better:
            best_metric = current
            best_model = model
            best_l2 = float(l2_val)
            if val_available:
                best_val_metrics, _, _ = _evaluate(
                    model,
                    val_feats,
                    val_labels,
                    num_classes,
                    label_names=label_names,
                    extra_metrics=extra_metrics,
                    avg_embeddings=avg_embeddings.get("val"),
                )
            else:
                best_val_metrics = monitor_metrics

        print(
            f"[Fold {fold_idx}] l2={l2_val:.2e} (C={C_val:.2e}) | val_{monitor_key}={current:.4f} | "
            f"best_l2={best_l2} ({best_metric:.4f}, minimize={minimize})"
        )

    if best_model is None:
        raise RuntimeError("Failed to fit any logistic regression model; check data inputs.")

    train_metrics, _, _ = _evaluate(
        best_model,
        train_feats,
        train_labels,
        num_classes,
        label_names=label_names,
        extra_metrics=extra_metrics,
        avg_embeddings=avg_embeddings.get("train"),
    )
    val_metrics = best_val_metrics
    test_metrics, _, _ = _evaluate(
        best_model,
        test_feats,
        test_labels,
        num_classes,
        label_names=label_names,
        extra_metrics=extra_metrics,
        avg_embeddings=avg_embeddings.get("test"),
    ) if test_feats is not None and test_labels is not None else ({}, None, None)

    fold_elapsed_s = max(0.0, float(time.time() - fold_t0))
    fold_elapsed_h = fold_elapsed_s / 3600.0
    try:
        writer.add_scalar("time/total_seconds", fold_elapsed_s, len(l2_grid))
        writer.add_scalar("time/total_hours", fold_elapsed_h, len(l2_grid))
        if best_l2 is not None:
            writer.add_scalar("logreg/selected_l2", best_l2, 0)
            writer.add_scalar("logreg/selected_C", 1.0 / best_l2, 0)
    except Exception:
        pass

    if isinstance(val_metrics, dict):
        val_metrics["train_time_hours"] = float(fold_elapsed_h)

    print(f"[Fold {fold_idx}] Best l2={best_l2} (C={1.0 / best_l2 if best_l2 else None}; monitor='{monitor_key}'={best_metric:.4f}; minimize={minimize})")
    if train_metrics:
        print("  Train:", {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in train_metrics.items()})
    if val_metrics:
        print("  Val:", {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in val_metrics.items()})
    if test_metrics:
        print("  Test:", {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in test_metrics.items()})
    print(f"  Timing: wall_time_hours={fold_elapsed_h:.4f} (≈ {fold_elapsed_s:.0f} s)")

    ckpt_root = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_dir = os.path.join(ckpt_root, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    try:
        import joblib

        joblib.dump(best_model, os.path.join(ckpt_dir, f"logreg_l2_{best_l2:.2e}.joblib"))
        with open(os.path.join(ckpt_dir, "selected_l2.txt"), "w") as f:
            f.write(str(best_l2))
    except Exception:
        pass

    fold_summary = {
        "fold": fold_idx,
        "monitor": monitor_key,
        "best_monitor_value": float(best_metric) if isinstance(best_metric, (int, float, np.floating)) else best_metric,
        "selected_l2": float(best_l2) if best_l2 is not None else None,
        "selected_C": float(1.0 / best_l2) if best_l2 else None,
        "l2_grid": [float(x) for x in l2_grid],
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "timing": {
            "start_time": datetime.fromtimestamp(fold_t0).isoformat(timespec="seconds"),
            "end_time": datetime.fromtimestamp(fold_t0 + fold_elapsed_s).isoformat(timespec="seconds"),
            "wall_time_seconds": float(fold_elapsed_s),
            "wall_time_hours": float(fold_elapsed_h),
        },
        "train_time_hours": float(fold_elapsed_h),
        "custom_metrics": list(extra_metrics),
    }

    try:
        with open(os.path.join(ckpt_dir, "best_metrics.json"), "w") as f:
            json.dump(fold_summary, f, indent=2)
    except Exception:
        pass

    writer.close()
    return fold_summary


def main():
    _configure_logging()
    args = parse_args()

    custom_metrics, sample_col, resolved_config_path = load_config_metrics(
        args.csv_path, config_path=args.config_path
    )
    args.custom_metrics = tuple(custom_metrics)
    args.sample_col = sample_col
    if custom_metrics:
        print(f"Detected custom metrics from config ({resolved_config_path}): {custom_metrics}")
    print(f"Using sample_col='{sample_col}' from config at {resolved_config_path}")

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
            sample_col=args.sample_col,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not fold_dfs:
        raise ValueError("No folds resolved from provided CSV arguments")
    print(
        f"Prepared {len(fold_dfs)} folds (split_seed={args.split_seed}; val≈1/{args.num_folds}; grouped by case_id when present) from {args.csv_path}"
    )

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
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                    "n_folds": len(vals),
                }
        return mean_std

    agg_train = _agg("train")
    agg_val = _agg("val")
    agg_test = _agg("test")

    try:
        fold_times_h = [
            float(res.get("timing", {}).get("wall_time_hours"))
            for res in all_fold_summaries
            if isinstance(res, dict)
            and isinstance(res.get("timing"), dict)
            and isinstance(res.get("timing", {}).get("wall_time_hours"), (int, float))
        ]
        if fold_times_h:
            total_h = float(np.sum(fold_times_h))
            agg_val = dict(agg_val)
            agg_val["train_time_hours"] = {
                "mean": float(np.mean(fold_times_h)),
                "std": float(np.std(fold_times_h, ddof=0)) if len(fold_times_h) > 1 else 0.0,
                "n_folds": len(fold_times_h),
            }
            agg_val["train_time_total_hours"] = {
                "mean": total_h,
                "std": None,
                "n_folds": len(fold_times_h),
            }
    except Exception:
        pass

    elapsed_s = max(0.0, float(time.time() - t0))
    elapsed_h = elapsed_s / 3600.0
    requested_folds = args.num_folds
    summary = {
        "monitor": args.monitor.split("/", 1)[-1],
        "num_folds": len(all_fold_summaries),
        "num_folds_requested": requested_folds,
        "split_seed": args.split_seed,
        "data_csv": args.csv_path,
        "folds": all_fold_summaries,
        "aggregate": {
            "train": agg_train,
            "val": agg_val,
            "test": agg_test,
        },
        "config_path": resolved_config_path,
        "custom_metrics": list(custom_metrics),
        "timing": {
            "start_time": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "end_time": datetime.fromtimestamp(t0 + elapsed_s).isoformat(timespec="seconds"),
            "wall_time_seconds": float(elapsed_s),
            "wall_time_hours": float(elapsed_h),
        },
    }

    try:
        out_summary_path = os.path.join(args.output_dir, "fold_summary.json")
        os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)
        with open(out_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
