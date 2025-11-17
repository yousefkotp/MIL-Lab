import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _index_feature_files(root: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    index: Dict[str, str] = {}
    for fp in files:
        basename = os.path.basename(fp)
        base_no_ext = os.path.splitext(basename)[0]
        index[base_no_ext] = fp
        index[basename] = fp
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
    ):
        super().__init__()
        if dataframe is not None:
            self.df = dataframe.copy()
        else:
            assert csv_path is not None, "csv_path or dataframe must be provided"
            self.df = pd.read_csv(csv_path)

        # Validate columns
        required_cols = ['filename', 'label', 'split']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}. Expected columns: 'filename', 'label', 'split'.")

        self.features_dir = features_dir
        self.allowed_exts = allowed_exts

        # Map labels to ints; preserve mapping if provided
        if '_y' in self.df.columns:
            pairs = self.df[['label', '_y']].drop_duplicates().sort_values('_y')
            self._label_to_idx = {row['label']: int(row['_y']) for _, row in pairs.iterrows()}
            self._idx_to_label = {int(row['_y']): row['label'] for _, row in pairs.iterrows()}
            self.df['_y'] = self.df['_y'].astype(int)
        else:
            self._label_to_idx = {l: i for i, l in enumerate(sorted(self.df['label'].unique()))}
            self._idx_to_label = {v: k for k, v in self._label_to_idx.items()}
            self.df['_y'] = self.df['label'].map(self._label_to_idx)

        self._file_index = _index_feature_files(self.features_dir, self.allowed_exts)
        keep_rows: List[bool] = []
        for _, row in self.df.iterrows():
            slide = str(row['filename']).strip()
            basename = os.path.basename(slide)
            base_no_ext = os.path.splitext(basename)[0]
            has_file = (slide in self._file_index) or (basename in self._file_index) or (base_no_ext in self._file_index)
            keep_rows.append(has_file)
        self.df = self.df[keep_rows].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(
                f"No feature files matched. Indexed {len(self._file_index)} files in {self.features_dir}."
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
        feats = _load_vector_features(fp)  # shape (D,)
        return feats, y, slide_id

