import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_SPLIT_SEED = 17


def _validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def _build_label_mapping(dfs: List[pd.DataFrame], label_col: str = 'label') -> Dict:
    concat = pd.concat(dfs, ignore_index=True)
    labels = sorted(concat[label_col].unique())
    return {label: idx for idx, label in enumerate(labels)}


def _apply_label_mapping(df: pd.DataFrame, label_to_idx: Dict, label_col: str = 'label') -> pd.DataFrame:
    out = df.copy()
    out['_y'] = out[label_col].map(label_to_idx)
    if out['_y'].isnull().any():
        missing = sorted(out.loc[out['_y'].isnull(), label_col].unique())
        raise ValueError(f"Could not map labels {missing}; provided mapping incomplete")
    out['_y'] = out['_y'].astype(int)
    return out


def _choose_splitter(num_folds: int, split_seed: int, groups: Optional[np.ndarray]):
    if groups is not None and StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=split_seed)
    if groups is not None and StratifiedGroupKFold is None:
        warnings.warn(
            "StratifiedGroupKFold is unavailable; falling back to StratifiedKFold without grouping.",
            RuntimeWarning,
        )
    return StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=split_seed)


def make_internal_folds(
    csv_path: str,
    num_folds: int,
    split_seed: int,
    *,
    label_col: str = 'label',
    filename_col: str = 'filename',
    group_col: str = 'case_id',
) -> Tuple[List[pd.DataFrame], Dict]:
    """Create stratified folds (optionally grouped by case_id) from a single CSV.

    The input CSV is expected to have columns: filename,label[,case_id]. A new
    'split' column is created per fold (train/val), and a consistent `_y` label
    index is attached so that all folds share identical label mappings.
    """
    df = pd.read_csv(csv_path)
    _validate_columns(df, [filename_col, label_col])

    if num_folds < 2:
        raise ValueError("num_folds must be at least 2 when generating folds")

    # Ingest once, drop any stale split indicator, and build a deterministic label map
    df = df.copy()
    if 'split' in df.columns:
        df = df.drop(columns=['split'])

    label_to_idx = _build_label_mapping([df], label_col=label_col)
    df = _apply_label_mapping(df, label_to_idx, label_col=label_col)

    y = df['_y'].to_numpy()
    groups = df[group_col].astype(str).to_numpy() if group_col in df.columns else None

    if groups is not None and len(np.unique(groups)) < num_folds:
        raise ValueError(
            f"num_folds={num_folds} exceeds number of unique groups ({len(np.unique(groups))})"
        )

    splitter = _choose_splitter(num_folds, split_seed, groups)
    try:
        if groups is not None:
            split_iter = splitter.split(np.zeros(len(df)), y, groups=groups)
        else:
            split_iter = splitter.split(np.zeros(len(df)), y)
    except Exception as e:
        raise RuntimeError(f"Failed to generate folds: {e}") from e

    fold_dfs: List[pd.DataFrame] = []
    for fold_idx, (_, val_idx) in enumerate(split_iter, start=1):
        fold_df = df.copy()
        fold_df['split'] = 'train'
        fold_df.loc[val_idx, 'split'] = 'val'
        fold_df['fold'] = fold_idx
        fold_dfs.append(fold_df)

    return fold_dfs, label_to_idx


def prepare_folds(
    *,
    csv_path: Optional[str] = None,
    num_folds: Optional[int] = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[pd.DataFrame], Dict]:
    """Resolve fold dataframes from a single unsplit CSV (preferred path).

    - Stratified by label; grouped by ``case_id`` when present so all slides from
      a case share the same split.
    - Produces train/val splits only; each fold holds roughly 1/``num_folds`` of
      the data as validation.
    """
    if not csv_path:
        raise ValueError("csv_path is required when generating folds internally")
    if num_folds is None:
        raise ValueError("num_folds must be set when using --csv_path")
    return make_internal_folds(csv_path, num_folds, split_seed)
