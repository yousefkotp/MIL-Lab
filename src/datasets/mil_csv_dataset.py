import os
import glob
import logging
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _index_feature_files(root: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    index = {}
    for fp in files:
        basename = os.path.basename(fp)
        base_no_ext = os.path.splitext(basename)[0]
        # map both without and with extension to improve matching robustness
        index[base_no_ext] = fp
        index[basename] = fp
    return index


def _row_major_sort_indices(coords_xy: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Compute indices that sort coordinates into row-major (y,x) order.

    Assumes coords are stored as (x, y) in shape (N, 2) or (2, N),
    consistent with the SlideHuBERT/AtlasPath extraction pipeline.

    Returns (indices, meta_info).
    """
    xy = np.asarray(coords_xy)
    if xy.ndim != 2:
        raise ValueError("coords must be a 2D array of shape (N,2) or (2,N)")
    if xy.shape[1] != 2 and xy.shape[0] == 2:
        xy = xy.T
    if xy.shape[1] != 2:
        raise ValueError(f"coords must have 2 columns; got shape {xy.shape}")

    x = xy[:, 0]
    y = xy[:, 1]
    idx = np.lexsort((x, y))  # primary sort by y, then x
    # Meta: unique grid extents (best-effort)
    H = int(len(np.unique(y)))
    W = int(len(np.unique(x)))
    meta = {
        'mapping': 'xy',
        'H': H,
        'W': W,
    }
    return idx, meta


def _load_features_sorted(fp: str) -> torch.Tensor:
    """Load features and coords from an HDF5 file and return features sorted in row-major.

    Assumes datasets are stored under keys 'features' and 'coords'. If not found, raises.
    """
    ext = os.path.splitext(fp)[1].lower()
    if ext not in ['.h5', '.hdf5']:
        raise ValueError(f"Unsupported feature file extension: {ext}. Only .h5/.hdf5 are supported.")
    import h5py
    with h5py.File(fp, 'r') as f:
        if 'features' not in f or 'coords' not in f:
            raise KeyError("Expected datasets 'features' and 'coords' in HDF5 file")
        feats = f['features'][()]
        coords = f['coords'][()]
        if feats.ndim != 2:
            raise ValueError(f"Expected 'features' with shape (N,D) or (D,), got {feats.shape}")
        # normalize coords to (N,2)
        if coords.ndim != 2:
            raise ValueError(f"Expected 'coords' 2D array, got {coords.shape}")
        if coords.shape[1] != 2:
            raise ValueError(f"Expected 'coords' shape (N,2) or (2,N), got {coords.shape}")
        if coords.shape[0] != feats.shape[0]:
            raise ValueError(f"Length mismatch: features N={feats.shape[0]} vs coords N={coords.shape[0]}")

        idx, meta = _row_major_sort_indices(coords)
        # Log minimal info for debugging without being verbose
        logger.debug(
            f"Sorted {os.path.basename(fp)} by row-major: mapping={meta.get('mapping')} "
            f"HxW={meta.get('H')}x{meta.get('W')}"
        )
        feats_sorted = feats[idx]
        return torch.from_numpy(feats_sorted).float()

class MILCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        features_dir: str = '',
        allowed_exts: Tuple[str, ...] = ('.h5', '.hdf5'),
        dataframe: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        if dataframe is not None:
            self.df = dataframe.copy()
        else:
            assert csv_path is not None, "csv_path or dataframe must be provided"
            self.df = pd.read_csv(csv_path)
        # Validate fixed schema: filename, label, split
        required_cols = ['filename', 'label', 'split']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}. Expected columns: 'filename', 'label', 'split'.")
        self.features_dir = features_dir
        self.allowed_exts = allowed_exts

        # Standardize labels to ints; if '_y' exists, preserve mapping across splits
        if '_y' in self.df.columns:
            pairs = self.df[['label', '_y']].drop_duplicates()
            # ensure one-to-one mapping per label
            pairs = pairs.sort_values('_y')
            self._label_to_idx = {row['label']: int(row['_y']) for _, row in pairs.iterrows()}
            self._idx_to_label = {int(row['_y']): row['label'] for _, row in pairs.iterrows()}
            # ensure df['_y'] is int dtype
            self.df['_y'] = self.df['_y'].astype(int)
        else:
            self._label_to_idx = {l: i for i, l in enumerate(sorted(self.df['label'].unique()))}
            self._idx_to_label = {v: k for k, v in self._label_to_idx.items()}
            self.df['_y'] = self.df['label'].map(self._label_to_idx)

        self._file_index = _index_feature_files(self.features_dir, self.allowed_exts)
        # filter out rows whose feature file is missing
        keep_rows: List[bool] = []
        missing_ids: List[str] = []
        for _, row in self.df.iterrows():
            slide = str(row['filename']).strip()
            basename = os.path.basename(slide)
            base_no_ext = os.path.splitext(basename)[0]
            has_file = (slide in self._file_index) or (basename in self._file_index) or (base_no_ext in self._file_index)
            keep_rows.append(has_file)
            if not has_file:
                missing_ids.append(slide)

        n_before = int(len(self.df))
        self.df = self.df[keep_rows].reset_index(drop=True)
        n_after = int(len(self.df))

        # Expose counters/ids for downstream consumers
        self.total_rows: int = n_before
        self.valid_rows: int = n_after
        self.missing_rows: int = n_before - n_after
        self.missing_slide_ids: List[str] = missing_ids

        if n_after == 0:
            raise RuntimeError(
                f"No feature files matched. Indexed {len(self._file_index)} files in {self.features_dir}."
            )
        # Log a concise summary instead of printing per-row
        logger.info(
            f"[MILCSVDataset] Matched {self.valid_rows}/{self.total_rows} rows with features; "
            f"missing={self.missing_rows} in {self.features_dir}"
        )

    @property
    def num_classes(self) -> int:
        return len(self._label_to_idx)

    @property
    def label_map(self) -> Dict[int, str]:
        return self._idx_to_label

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        slide_id = str(row['filename'])
        y = int(row['_y'])
        base_no_ext = os.path.splitext(os.path.basename(slide_id))[0]
        basename = os.path.basename(slide_id)
        fp = self._file_index.get(slide_id) or self._file_index.get(basename) or self._file_index.get(base_no_ext)
        if fp is None:
            raise FileNotFoundError(f"No features for slide_id={slide_id}")
        # Always load features and sort them by row-major (y,x) using coords
        feats = _load_features_sorted(fp)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return feats, y, slide_id
