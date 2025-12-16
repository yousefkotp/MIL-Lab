import argparse
import os
import sys
import pandas as pd

from src.metrics_utils import load_config_metrics
from src.datasets.mil_csv_dataset import MILCSVDataset
from src.datasets.linear_csv_dataset import LinearCSVDataset
from src.datasets.knn_csv_dataset import KNNCSVDataset
from src.datasets.logreg_csv_dataset import LogRegCSVDataset

def build_manifest(csv_path: str, config_path: str, features_dir: str, parent_dir: str, mode: str) -> list[str]:
    """Return list of lines '<abs_path>\\t<rel_path>' for matching feature files."""
    if mode not in ("mil", "linear", "knn"):
        raise ValueError(f"mode must be 'mil', 'linear', or 'knn', got {mode}")
    if mode == "mil":
        ds_cls = MILCSVDataset
    elif mode == "knn":
        ds_cls = KNNCSVDataset
    elif mode == "linear":
        ds_cls = LinearCSVDataset
    else:
        ds_cls = LogRegCSVDataset

    metrics, sample_col, _ = load_config_metrics(csv_path=csv_path, config_path=config_path)
    _ = metrics  # unused; kept for future extensions

    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        df = df.copy()
        df["split"] = "train"

    ds = ds_cls(csv_path=None, dataframe=df, features_dir=features_dir, sample_col=sample_col, feature_parent_dir=parent_dir)
    paths = sorted({os.path.abspath(p) for s in ds.samples for p in s.get("paths", [])})
    if not paths:
        raise RuntimeError("No matching feature files found for requested samples.")

    root = os.path.abspath(os.path.expanduser(features_dir))
    lines: list[str] = []
    for p in paths:
        common = os.path.commonpath([root, p])
        if common != root:
            raise RuntimeError(f"Feature path {p} is outside features_dir {root}")
        rel = os.path.relpath(p, root)
        lines.append(f"{p}\t{rel}")
    return lines


def parse_args():
    ap = argparse.ArgumentParser(description="Build manifest of feature files required for a task CSV.")
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--parent_dir", required=True, help="Name of the parent directory containing .h5 files")
    ap.add_argument("--mode", choices=["mil", "linear", "knn", "logreg"], required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        lines = build_manifest(
            csv_path=args.csv_path,
            config_path=args.config_path,
            features_dir=args.features_dir,
            parent_dir=args.parent_dir,
            mode=args.mode,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"{exc}\n")
        sys.exit(1)

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
