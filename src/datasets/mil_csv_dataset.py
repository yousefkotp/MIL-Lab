import os
import logging
from typing import List, Optional, Dict, Tuple, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _index_feature_files(root: str, exts: Tuple[str, ...], target_stems: Set[str]) -> Dict[str, str]:
    """Index feature files under ``root`` only for the requested stems.

    Stems are the basename without extension from the CSV (e.g., `patient_001_node_0`).
    Search is recursive so embeddings can live in nested subdirectories.
    """
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"features_dir '{root}' does not exist or is not a directory")

    stems_lower = {s.lower() for s in target_stems}
    index: Dict[str, str] = {}
    lower_exts = tuple(ext.lower() for ext in exts)
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(lower_exts):
                continue
            fp = os.path.join(dirpath, fname)
            basename = os.path.basename(fp)
            base_no_ext = os.path.splitext(basename)[0]
            if base_no_ext.lower() not in stems_lower:
                continue
            index[base_no_ext] = fp  # stem key
            index[basename] = fp     # embedding basename (e.g., .h5)
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


def _as_list(val) -> List[str]:
    """Coerce a value into a list of strings."""
    if isinstance(val, (list, tuple, set)):
        return [str(v) for v in val]
    if pd.isna(val):
        return []
    return [str(val)]


class MILCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        features_dir: str = '',
        allowed_exts: Tuple[str, ...] = ('.h5', '.hdf5'),
        dataframe: Optional[pd.DataFrame] = None,
        case_fusion: str = 'late',
    ):
        """
        Dataset that loads patch-level features and supports multiple slides per case.

        If a ``case_id`` column is present (or ``slide_ids`` lists are provided), rows are
        grouped by case. ``case_fusion`` controls how slides are combined:
          - 'late' (default): keep slides separate and average logits in the training loop.
          - 'early': concatenate patch embeddings from all slides into one bag.
        """
        super().__init__()
        if case_fusion not in ('late', 'early'):
            raise ValueError(f"case_fusion must be 'late' or 'early', got {case_fusion}")
        self.case_fusion = case_fusion

        if dataframe is not None:
            df = dataframe.copy()
        else:
            assert csv_path is not None, "csv_path or dataframe must be provided"
            df = pd.read_csv(csv_path)

        # Validate presence of core columns (allow grouped data with slide_ids)
        required_cols = ['label', 'split']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}. Expected at least: {required_cols}")
        if 'filename' not in df.columns and 'slide_ids' not in df.columns:
            raise ValueError("CSV must contain either 'filename' (per-slide) or 'slide_ids' (list per case).")

        self.features_dir = os.path.abspath(os.path.expanduser(features_dir))
        self.allowed_exts = allowed_exts

        # Label mapping (preserve if provided)
        if '_y' in df.columns:
            pairs = df[['label', '_y']].drop_duplicates().sort_values('_y')
            self._label_to_idx = {row['label']: int(row['_y']) for _, row in pairs.iterrows()}
            self._idx_to_label = {int(row['_y']): row['label'] for _, row in pairs.iterrows()}
            df['_y'] = df['_y'].astype(int)
        else:
            self._label_to_idx = {l: i for i, l in enumerate(sorted(df['label'].unique()))}
            self._idx_to_label = {v: k for k, v in self._label_to_idx.items()}
            df['_y'] = df['label'].map(self._label_to_idx)

        self.raw_df = df.copy()

        # Build per-sample (case) records
        self.samples = []
        if 'slide_ids' in df.columns:
            for _, row in df.iterrows():
                slide_ids = _as_list(row['slide_ids'])
                if len(slide_ids) == 0:
                    continue
                case_id = row['case_id'] if 'case_id' in row else None
                sample_id = str(case_id) if case_id is not None and not (isinstance(case_id, float) and np.isnan(case_id)) else slide_ids[0]
                split = row['split'] if 'split' in row else 'train'
                self.samples.append({
                    'id': str(sample_id),
                    'case_id': str(case_id) if case_id is not None else None,
                    'slide_ids': slide_ids,
                    'label': row['label'],
                    'label_idx': int(row['_y']),
                    'split': split,
                })
        elif 'case_id' in df.columns:
            grouped = df.groupby(df['case_id'].astype(str), sort=False)
            for case_id, grp in grouped:
                slide_ids = grp['filename'].astype(str).tolist()
                if len(slide_ids) == 0:
                    continue
                label_vals = grp['label'].unique()
                if len(label_vals) != 1:
                    raise ValueError(f"Inconsistent labels for case_id={case_id}: {label_vals}")
                y_vals = grp['_y'].unique()
                if len(y_vals) != 1:
                    raise ValueError(f"Inconsistent mapped labels for case_id={case_id}: {y_vals}")
                split_vals = grp['split'].unique() if 'split' in grp.columns else ['train']
                if len(split_vals) != 1:
                    raise ValueError(f"Inconsistent split assignments for case_id={case_id}: {split_vals}")
                self.samples.append({
                    'id': str(case_id),
                    'case_id': str(case_id),
                    'slide_ids': slide_ids,
                    'label': label_vals[0],
                    'label_idx': int(y_vals[0]),
                    'split': split_vals[0],
                })
        else:
            for _, row in df.iterrows():
                slide_id = str(row['filename'])
                if not slide_id:
                    continue
                split = row['split'] if 'split' in row else 'train'
                self.samples.append({
                    'id': slide_id,
                    'case_id': None,
                    'slide_ids': [slide_id],
                    'label': row['label'],
                    'label_idx': int(row['_y']),
                    'split': split,
                })

        # Index feature files for all slides we expect
        all_slide_ids: List[str] = []
        for s in self.samples:
            all_slide_ids.extend(s['slide_ids'])
        target_stems: Set[str] = set()
        for slide in all_slide_ids:
            slide = str(slide).strip()
            base = os.path.basename(slide)
            base_no_ext = os.path.splitext(base)[0]
            target_stems.add(base_no_ext)

        self._file_index = _index_feature_files(self.features_dir, self.allowed_exts, target_stems=target_stems)

        # Filter out samples whose slides are missing
        filtered_samples: List[dict] = []
        missing_ids: List[str] = []
        matched_slides = 0
        for sample in self.samples:
            paths = []
            resolved_slide_ids = []
            for slide in sample['slide_ids']:
                basename = os.path.basename(str(slide).strip())
                base_no_ext = os.path.splitext(basename)[0]
                if base_no_ext in self._file_index:
                    self._file_index.setdefault(basename, self._file_index[base_no_ext])
                    self._file_index.setdefault(slide, self._file_index[base_no_ext])
                fp = self._file_index.get(basename) or self._file_index.get(base_no_ext) or self._file_index.get(slide)
                if fp is None:
                    missing_ids.append(str(slide))
                    continue
                paths.append(fp)
                resolved_slide_ids.append(basename)
            if len(paths) == 0:
                continue
            matched_slides += len(paths)
            filtered_samples.append({**sample, 'paths': paths, 'slide_ids': resolved_slide_ids})

        total_slides = len(all_slide_ids)
        self.samples = filtered_samples
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No feature files matched. Indexed {len(self._file_index)} files in {self.features_dir}."
            )

        # Aggregated dataframe aligned with dataset length (one row per case/sample)
        self.df = pd.DataFrame([{
            'id': s['id'],
            'case_id': s.get('case_id'),
            'filename': s['slide_ids'][0],  # first slide as representative
            'slide_ids': s['slide_ids'],
            'split': s['split'],
            'label': s['label'],
            '_y': s['label_idx'],
            'num_slides': len(s['slide_ids']),
        } for s in self.samples])

        # Expose counters/ids for downstream consumers
        self.total_rows: int = total_slides
        self.valid_rows: int = matched_slides
        self.missing_rows: int = total_slides - matched_slides
        self.missing_slide_ids: List[str] = missing_ids

        logger.info(
            f"[MILCSVDataset] Using {len(self.samples)} samples (grouped by {'case' if 'case_id' in df.columns else 'slide'}) "
            f"with case_fusion={self.case_fusion}. Matched {self.valid_rows}/{self.total_rows} slides; missing={self.missing_rows}."
        )

    @property
    def num_classes(self) -> int:
        return len(self._label_to_idx)

    @property
    def label_map(self) -> Dict[int, str]:
        return self._idx_to_label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        y = int(sample['label_idx'])
        slide_paths = sample.get('paths') or []
        if len(slide_paths) == 0:
            raise FileNotFoundError(f"No features for sample id={sample['id']}")

        slide_feats: List[torch.Tensor] = []
        for fp in slide_paths:
            feats = _load_features_sorted(fp)
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            slide_feats.append(feats)

        if self.case_fusion == 'early' and len(slide_feats) > 1:
            feats_out = torch.cat(slide_feats, dim=0)
        else:
            feats_out = slide_feats if len(slide_feats) > 1 else slide_feats[0]

        return feats_out, y, sample['id']
