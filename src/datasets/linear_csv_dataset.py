import os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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


def _load_vector_features(fp: str) -> torch.Tensor:
    """Load 1D vector features from an HDF5 file.

    Expects dataset 'features' shaped (D,) or (1, D). Returns a float tensor of shape (D,).
    """
    ext = os.path.splitext(fp)[1].lower()
    if ext not in ['.h5', '.hdf5']:
        raise ValueError(f"Unsupported feature file extension: {ext}. Only .h5/.hdf5 are supported.")
    import h5py
    with h5py.File(fp, 'r') as f:
        if 'features' not in f:
            raise KeyError("Expected dataset 'features' in HDF5 file")
        feats = f['features'][()]
        arr = np.asarray(feats)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"Expected vector features with shape (D,) or (1,D); got {arr.shape}")
        return torch.from_numpy(arr).float()


def _as_list(val) -> List[str]:
    """Coerce a value into a list of strings."""
    if isinstance(val, (list, tuple, set)):
        return [str(v) for v in val]
    if pd.isna(val):
        return []
    return [str(val)]


class LinearCSVDataset(Dataset):
    """Dataset for linear probe training with per-WSI vector features.

    CSV schema: filename,label,split
    Features are stored in HDF5 files under 'features' with shape (D,) or (1,D).
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        features_dir: str = '',
        allowed_exts: Tuple[str, ...] = ('.h5', '.hdf5'),
        dataframe: Optional[pd.DataFrame] = None,
        sample_col: str = 'case_id',
        case_fusion: str = 'late',
    ):
        super().__init__()
        if case_fusion not in ('late', 'early'):
            raise ValueError(f"case_fusion must be 'late' or 'early', got {case_fusion}")
        self.case_fusion = case_fusion
        sample_col_norm = str(sample_col).strip().lower()
        if sample_col_norm not in ('filename', 'case_id'):
            raise ValueError(f"sample_col must be 'filename' or 'case_id', got {sample_col}")
        self.sample_col = sample_col_norm

        if dataframe is not None:
            df = dataframe.copy()
        else:
            assert csv_path is not None, "csv_path or dataframe must be provided"
            df = pd.read_csv(csv_path)

        # Validate columns
        required_cols = ['label', 'split', self.sample_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}. Expected at least: {required_cols}")
        if 'filename' not in df.columns and 'slide_ids' not in df.columns:
            raise ValueError("CSV must contain either 'filename' (per-slide) or 'slide_ids' (list per case).")
        if self.sample_col == 'case_id' and df['case_id'].isnull().any():
            raise ValueError("sample_col='case_id' selected but 'case_id' column contains missing values.")

        self.features_dir = os.path.abspath(os.path.expanduser(features_dir))
        self.allowed_exts = allowed_exts

        # Map labels to ints; preserve mapping if provided
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
        # Ensure each sample (per sample_col) maps to exactly one label
        dup_mask = (
            df.groupby(df[self.sample_col].astype(str))['_y']
            .nunique(dropna=False)
            .reset_index()
        )
        bad = dup_mask[dup_mask['_y'] > 1][self.sample_col].tolist()
        if bad:
            raise ValueError(f"Inconsistent labels for {self.sample_col} values: {bad[:5]} (and {max(0, len(bad)-5)} more)")

        # Build per-sample records based on sample_col policy
        self.samples = []
        if 'slide_ids' in df.columns:
            for _, row in df.iterrows():
                slide_ids = _as_list(row['slide_ids'])
                if len(slide_ids) == 0:
                    continue
                split = row['split'] if 'split' in row else 'train'
                if self.sample_col == 'case_id':
                    case_id = row['case_id'] if 'case_id' in row else None
                    sample_id = case_id
                    if sample_id is None or (isinstance(sample_id, float) and np.isnan(sample_id)):
                        sample_id = slide_ids[0]
                    self.samples.append({
                        'id': str(sample_id),
                        'case_id': str(case_id) if case_id is not None else None,
                        'slide_ids': slide_ids,
                        'label': row['label'],
                        'label_idx': int(row['_y']),
                        'split': split,
                    })
                else:  # sample_col == 'filename'
                    case_id_val = row['case_id'] if 'case_id' in row else None
                    for slide_id in slide_ids:
                        sid = str(slide_id)
                        self.samples.append({
                            'id': sid,
                            'case_id': str(case_id_val) if case_id_val is not None and not (isinstance(case_id_val, float) and np.isnan(case_id_val)) else None,
                            'slide_ids': [sid],
                            'label': row['label'],
                            'label_idx': int(row['_y']),
                            'split': split,
                        })
        elif self.sample_col == 'case_id':
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
                case_id_val = row['case_id'] if 'case_id' in row else None
                self.samples.append({
                    'id': slide_id,
                    'case_id': str(case_id_val) if case_id_val is not None and not (isinstance(case_id_val, float) and np.isnan(case_id_val)) else None,
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
            'filename': s['slide_ids'][0],
            'slide_ids': s['slide_ids'],
            'split': s['split'],
            'label': s['label'],
            '_y': s['label_idx'],
            'num_slides': len(s['slide_ids']),
        } for s in self.samples])

        self.total_rows: int = total_slides
        self.valid_rows: int = matched_slides
        self.missing_rows: int = total_slides - matched_slides
        self.missing_slide_ids: List[str] = missing_ids

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
            feats = _load_vector_features(fp)  # shape (D,)
            slide_feats.append(feats)

        if len(slide_feats) == 1:
            fused = slide_feats[0]
        else:
            stacked = torch.stack(slide_feats, dim=0)
            # Mean of per-slide embeddings -> late fusion at embedding level
            fused = stacked.mean(dim=0)

        return fused, y, sample['id']
