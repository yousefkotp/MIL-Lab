import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

# Aliases map to canonical metric keys that will be emitted in summaries
_METRIC_ALIASES: Dict[str, str] = {
    'weighted_kappa': 'weighted_kappa',
    'quadratic_kappa': 'weighted_kappa',
    'cohen_kappa': 'weighted_kappa',
}
_WARNED_UNKNOWN: set = set()


def _normalize_metric_name(name: str) -> str:
    """Normalize metric names to lowercase snake_case (alnum + underscores)."""
    norm = str(name).strip().lower().replace(' ', '_')
    norm = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in norm)
    while '__' in norm:
        norm = norm.replace('__', '_')
    return norm.strip('_')


def load_config_metrics(csv_path: str, config_path: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """Load optional custom metrics from a task config.yaml located next to the CSV.

    Returns (metric_names, resolved_config_path). Metric names are normalized.
    Missing or unreadable config files yield an empty list.
    """
    resolved_cfg = config_path
    if resolved_cfg is None and csv_path:
        resolved_cfg = os.path.join(os.path.dirname(os.path.abspath(csv_path)), 'config.yaml')

    if not resolved_cfg or not os.path.isfile(resolved_cfg):
        return [], resolved_cfg

    try:
        with open(resolved_cfg, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        logger.debug("Failed to read config.yaml at %s", resolved_cfg, exc_info=True)
        return [], resolved_cfg

    raw_metrics = cfg.get('metrics', [])
    if isinstance(raw_metrics, (list, tuple, set)):
        items: Iterable = raw_metrics
    else:
        items = [raw_metrics]

    metric_names: List[str] = []
    seen: set = set()
    for item in items:
        if item is None:
            continue
        name = _normalize_metric_name(item)
        if not name or name in seen:
            continue
        seen.add(name)
        # Keep canonical alias names only
        canonical = _METRIC_ALIASES.get(name, name)
        metric_names.append(canonical)
    return metric_names, resolved_cfg


def compute_additional_metrics(metric_names: Sequence[str], labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Compute optional metrics requested in config.yaml.

    Currently supports:
      - weighted_kappa (quadratic-weighted Cohen's kappa)
    Unknown metrics are skipped after a single warning.
    """
    extras: Dict[str, float] = {}
    if labels is None or preds is None:
        return extras

    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    seen: set = set()
    for name in metric_names:
        canonical = _METRIC_ALIASES.get(_normalize_metric_name(name), _normalize_metric_name(name))
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        try:
            if canonical == 'weighted_kappa':
                extras[canonical] = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
            else:
                if canonical not in _WARNED_UNKNOWN:
                    logger.warning("Requested custom metric '%s' is not implemented; skipping.", canonical)
                    _WARNED_UNKNOWN.add(canonical)
        except Exception:
            extras[canonical] = float('nan')
    return extras
