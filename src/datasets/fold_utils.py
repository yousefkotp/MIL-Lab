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


def _assert_single_label_per_sample(df: pd.DataFrame, sample_col: str, label_col: str = '_y') -> None:
    """Ensure each sample (per sample_col) has a unique label; else raise."""
    if sample_col not in df.columns:
        raise ValueError(f"sample_col '{sample_col}' not found in dataframe columns")
    dup_mask = (
        df.groupby(df[sample_col].astype(str))[label_col]
        .nunique(dropna=False)
        .reset_index()
    )
    bad = dup_mask[dup_mask[label_col] > 1][sample_col].tolist()
    if bad:
        raise ValueError(f"Inconsistent labels for {sample_col} values: {bad[:5]} (and {max(0, len(bad)-5)} more)")


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
    sample_col: str,
    label_col: str = 'label',
    filename_col: str = 'filename',
) -> Tuple[List[pd.DataFrame], Dict]:
    """Create stratified folds from a single CSV with configurable sampling unit.

    The input CSV is expected to have columns: filename,label[,case_id]. A new
    'split' column is created per fold (train/val), and a consistent `_y` label
    index is attached so that all folds share identical label mappings.
    Grouping is controlled by ``sample_col`` ('case_id' -> StratifiedGroupKFold,
    'filename' -> StratifiedKFold without grouping).
    """
    df = pd.read_csv(csv_path)
    sample_col_norm = str(sample_col).strip().lower()
    if sample_col_norm not in ('filename', 'case_id'):
        raise ValueError(f"sample_col must be 'filename' or 'case_id'; got '{sample_col}'")

    required_cols = [label_col]
    if filename_col:
        required_cols.append(filename_col)
    if sample_col_norm == 'case_id':
        required_cols.append('case_id')
    _validate_columns(df, required_cols)

    if num_folds < 2:
        raise ValueError("num_folds must be at least 2 when generating folds")

    # Ingest once, drop any stale split indicator, and build a deterministic label map
    df = df.copy()
    if 'split' in df.columns:
        df = df.drop(columns=['split'])

    label_to_idx = _build_label_mapping([df], label_col=label_col)
    df = _apply_label_mapping(df, label_to_idx, label_col=label_col)
    _assert_single_label_per_sample(df, sample_col=sample_col_norm, label_col='_y')

    y = df['_y'].to_numpy()
    groups = None
    if sample_col_norm == 'case_id':
        if df['case_id'].isnull().any():
            missing = sorted(df.loc[df['case_id'].isnull(), 'case_id'].unique())
            raise ValueError(f"sample_col=case_id selected but case_id contains missing values: {missing}")
        groups = df['case_id'].astype(str).to_numpy()

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
    sample_col: str,
) -> Tuple[List[pd.DataFrame], Dict]:
    """Resolve fold dataframes from a single unsplit CSV (preferred path).

    - Stratified by label; grouping controlled by ``sample_col`` ('case_id' or 'filename').
    - Produces train/val splits only; each fold holds roughly 1/``num_folds`` of
      the data as validation.
    """
    if not csv_path:
        raise ValueError("csv_path is required when generating folds internally")
    if num_folds is None:
        raise ValueError("num_folds must be set when using --csv_path")
    return make_internal_folds(csv_path, num_folds, split_seed, sample_col=sample_col)
