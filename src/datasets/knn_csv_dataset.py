import torch
import torch.nn.functional as F

from src.datasets.linear_csv_dataset import LinearCSVDataset, _load_vector_features


class KNNCSVDataset(LinearCSVDataset):
    """Dataset for k-NN that enforces L2 on slides and fused case embeddings when requested."""

    def __init__(self, *args, l2_normalize: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_normalize = bool(l2_normalize)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        y = int(sample["label_idx"])
        slide_paths = sample.get("paths") or []
        if len(slide_paths) == 0:
            raise FileNotFoundError(f"No features for sample id={sample['id']}")

        slide_feats = []
        for fp in slide_paths:
            feats = _load_vector_features(fp).float()  # shape (D,)
            if self.l2_normalize:
                feats = F.normalize(feats, p=2, dim=0, eps=1e-8)
            slide_feats.append(feats)

        if len(slide_feats) == 1:
            fused = slide_feats[0]
        else:
            stacked = torch.stack(slide_feats, dim=0)
            fused = stacked.mean(dim=0)
        if self.l2_normalize:
            fused = F.normalize(fused, p=2, dim=0, eps=1e-8)

        return fused, y, sample["id"]
