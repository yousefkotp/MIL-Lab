#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import subprocess
import shutil


@dataclass
class ValAggregate:
    mean: float
    std: Optional[float] = None
    n_folds: Optional[int] = None


def _is_number(x) -> bool:
    try:
        return isinstance(x, (int, float)) and not (isinstance(x, float) and (x != x))
    except Exception:
        return False


def _find_fold_summary_file(method_dir: str) -> Optional[str]:
    """Find a fold_summary.json inside a method directory (robust to nesting)."""
    candidate = os.path.join(method_dir, 'fold_summary.json')
    if os.path.isfile(candidate):
        return candidate
    for root, _, files in os.walk(method_dir):
        if 'fold_summary.json' in files:
            return os.path.join(root, 'fold_summary.json')
    return None


def _load_val_aggregate(summary_path: str) -> Dict[str, ValAggregate]:
    """Load validation aggregate metrics from a fold_summary.json.

    If the file contains precomputed aggregate['val'], use that.
    Otherwise, compute means across folds from folds[i]['val'] numeric entries.
    """
    with open(summary_path, 'r') as f:
        data = json.load(f)

    # Preferred: precomputed aggregate
    agg_val = None
    if isinstance(data, dict):
        agg = data.get('aggregate') or {}
        agg_val = agg.get('val') if isinstance(agg, dict) else None

    if isinstance(agg_val, dict) and len(agg_val) > 0:
        out: Dict[str, ValAggregate] = {}
        for k, v in agg_val.items():
            if isinstance(v, dict) and 'mean' in v:
                mean = v.get('mean')
                std = v.get('std')
                n = v.get('n_folds') or v.get('n')
                if _is_number(mean):
                    out[k] = ValAggregate(mean=float(mean), std=float(std) if _is_number(std) else None, n_folds=int(n) if isinstance(n, int) else None)
        # Also try to attach training time metrics if present elsewhere (fold timing)
        try:
            folds = data.get('folds') if isinstance(data, dict) else None
            fold_times: List[float] = []
            if isinstance(folds, list):
                for f in folds:
                    if not isinstance(f, dict):
                        continue
                    tblock = f.get('timing') if isinstance(f.get('timing'), dict) else None
                    tv = None
                    if tblock and _is_number(tblock.get('wall_time_hours')):
                        tv = float(tblock.get('wall_time_hours'))
                    elif _is_number(f.get('train_time_hours')):
                        tv = float(f.get('train_time_hours'))
                    if tv is not None:
                        fold_times.append(tv)
            if fold_times:
                import statistics
                mean_t = sum(fold_times) / len(fold_times)
                std_t = statistics.pstdev(fold_times) if len(fold_times) > 1 else 0.0
                if 'train_time_hours' not in out:
                    out['train_time_hours'] = ValAggregate(mean=mean_t, std=std_t, n_folds=len(fold_times))
                total_t = sum(fold_times)
                if 'train_time_total_hours' not in out:
                    out['train_time_total_hours'] = ValAggregate(mean=total_t, std=None, n_folds=len(fold_times))
        except Exception:
            pass
        if out:
            return out

    # Fallback: compute from folds
    folds = data.get('folds') if isinstance(data, dict) else None
    values: Dict[str, List[float]] = {}
    if isinstance(folds, list):
        for fold in folds:
            val = fold.get('val') if isinstance(fold, dict) else None
            if not isinstance(val, dict):
                continue
            for k, v in val.items():
                if _is_number(v):
                    values.setdefault(k, []).append(float(v))

    from math import fsum
    import statistics
    out: Dict[str, ValAggregate] = {}
    for k, vs in values.items():
        if not vs:
            continue
        mean = fsum(vs) / len(vs)
        std = statistics.pstdev(vs) if len(vs) > 1 else 0.0
        out[k] = ValAggregate(mean=mean, std=std, n_folds=len(vs))

    # If timing info exists per-fold but not captured above, attach it too
    try:
        folds = data.get('folds') if isinstance(data, dict) else None
        fold_times: List[float] = []
        if isinstance(folds, list):
            for f in folds:
                if not isinstance(f, dict):
                    continue
                tblock = f.get('timing') if isinstance(f.get('timing'), dict) else None
                tv = None
                if tblock and _is_number(tblock.get('wall_time_hours')):
                    tv = float(tblock.get('wall_time_hours'))
                elif _is_number(f.get('train_time_hours')):
                    tv = float(f.get('train_time_hours'))
                if tv is not None:
                    fold_times.append(tv)
        if fold_times:
            mean_t = fsum(fold_times) / len(fold_times)
            std_t = statistics.pstdev(fold_times) if len(fold_times) > 1 else 0.0
            if 'train_time_hours' not in out:
                out['train_time_hours'] = ValAggregate(mean=mean_t, std=std_t, n_folds=len(fold_times))
            total_t = fsum(fold_times)
            if 'train_time_total_hours' not in out:
                out['train_time_total_hours'] = ValAggregate(mean=total_t, std=None, n_folds=len(fold_times))
    except Exception:
        pass

    return out


def _list_method_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    items = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and not name.startswith('.'):
            items.append(path)
    return items


def collect_root_results(root: str) -> Dict[str, Dict[str, ValAggregate]]:
    """Return mapping: method_name -> { metric -> ValAggregate } for a root path.

    Typical MIL roots have per-method subfolders containing a fold_summary.json.
    Linear runs may place a single fold_summary.json directly at the root.
    This function handles both cases.
    """
    results: Dict[str, Dict[str, ValAggregate]] = {}

    # Special case: a fold_summary.json directly at the provided root
    root_level_summary = os.path.join(root, 'fold_summary.json')
    if os.path.isfile(root_level_summary):
        # Name the method based on path. If this is a linear path, prefer 'no_mil'.
        base = os.path.basename(os.path.normpath(root)) or 'root'
        method_name = 'no_mil' if ('linear' in root.lower()) else base
        try:
            results[method_name] = _load_val_aggregate(root_level_summary)
        except Exception:
            pass

    # Standard case: subdirectories per method
    for method_dir in _list_method_dirs(root):
        method = os.path.basename(method_dir)
        summary_path = _find_fold_summary_file(method_dir)
        if not summary_path:
            continue
        try:
            results[method] = _load_val_aggregate(summary_path)
        except Exception:
            # Skip unreadable/invalid JSONs
            continue
    return results


def compare_roots(root_a: str, root_b: str) -> Tuple[Dict[str, Dict[str, ValAggregate]], Dict[str, Dict[str, ValAggregate]]]:
    a = collect_root_results(root_a)
    b = collect_root_results(root_b)
    return a, b


def collect_roots_multi(roots: List[str]) -> List[Dict[str, Dict[str, ValAggregate]]]:
    """Collect results for multiple roots.

    Returns a list parallel to `roots`, where each item is a mapping:
    method_name -> { metric -> ValAggregate }.
    """
    out: List[Dict[str, Dict[str, ValAggregate]]] = []
    for r in roots:
        out.append(collect_root_results(r))
    return out


def format_float(x: Optional[float], places: int = 4, keep_trailing: bool = False) -> str:
    if x is None:
        return 'NA'
    fmt = f"{{:.{places}f}}"
    s = fmt.format(x)
    return s if keep_trailing else s.rstrip('0').rstrip('.')


def _pick_nfolds(metrics: Dict[str, ValAggregate]) -> Optional[int]:
    for v in metrics.values():
        if v and isinstance(v.n_folds, int):
            return v.n_folds
    return None


def _build_rows(method: str,
                a_metrics: Dict[str, ValAggregate],
                b_metrics: Dict[str, ValAggregate],
                places: int = 4) -> List[Tuple[str, str, str, str]]:
    all_metrics = set(a_metrics.keys()) | set(b_metrics.keys())
    rows: List[Tuple[str, str, str, str]] = []
    for metric in all_metrics:
        va = a_metrics.get(metric)
        vb = b_metrics.get(metric)
        ma = va.mean if va else None
        mb = vb.mean if vb else None
        sa = va.std if va else None
        sb = vb.std if vb else None
        delta = (mb - ma) if (ma is not None and mb is not None) else None
        a_str = 'NA' if ma is None else (
            f"{format_float(ma, places, keep_trailing=True)} ± {format_float(sa, places, keep_trailing=True)}" if sa is not None else f"{format_float(ma, places, keep_trailing=True)}"
        )
        b_str = 'NA' if mb is None else (
            f"{format_float(mb, places, keep_trailing=True)} ± {format_float(sb, places, keep_trailing=True)}" if sb is not None else f"{format_float(mb, places, keep_trailing=True)}"
        )
        d_str = format_float(delta, places, keep_trailing=True)
        rows.append((metric, a_str, b_str, d_str))

    # Always sort metrics by name
    rows.sort(key=lambda r: r[0])
    return rows


def _print_table(method: str,
                 root_a_name: str,
                 root_b_name: str,
                 a_metrics: Dict[str, ValAggregate],
                 b_metrics: Dict[str, ValAggregate],
                 places: int = 4):
    n_a = _pick_nfolds(a_metrics)
    n_b = _pick_nfolds(b_metrics)
    subtitle = f"{root_a_name} (n={n_a}) vs {root_b_name} (n={n_b})" if (n_a or n_b) else f"{root_a_name} vs {root_b_name}"
    print(f"\n=== {method} ===")
    print(subtitle)

    rows = _build_rows(method, a_metrics, b_metrics, places=places)
    # headers
    headers = (
        'Metric',
        f"{root_a_name} (mean ± std)",
        f"{root_b_name} (mean ± std)",
        'Δ (B−A)'
    )
    # compute column widths
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(str(x)) for x in col) for col in cols]
    # clamp metric column width to a reasonable size
    widths[0] = min(widths[0], 28)

    def trunc(s: str, w: int) -> str:
        return s if len(s) <= w else (s[: w - 1] + '…')

    def hline(ch: str = '-') -> str:
        return '+' + '+'.join(ch * (w + 2) for w in widths) + '+'

    print(hline('-'))
    print('| ' + ' | '.join(
        trunc(headers[i], widths[i]).ljust(widths[i]) if i == 0 else trunc(headers[i], widths[i]).rjust(widths[i])
        for i in range(len(headers))
    ) + ' |')
    print(hline('='))
    for m, a_str, b_str, d_str in rows:
        print('| ' + ' | '.join([
            trunc(m, widths[0]).ljust(widths[0]),
            trunc(a_str, widths[1]).rjust(widths[1]),
            trunc(b_str, widths[2]).rjust(widths[2]),
            trunc(d_str, widths[3]).rjust(widths[3]),
        ]) + ' |')
    print(hline('-'))


def _print_markdown(method: str,
                    root_a_name: str,
                    root_b_name: str,
                    a_metrics: Dict[str, ValAggregate],
                    b_metrics: Dict[str, ValAggregate],
                    places: int = 4):
    n_a = _pick_nfolds(a_metrics)
    n_b = _pick_nfolds(b_metrics)
    subtitle = f"{root_a_name} (n={n_a}) vs {root_b_name} (n={n_b})" if (n_a or n_b) else f"{root_a_name} vs {root_b_name}"
    print(f"\n### {method}")
    print(subtitle)
    rows = _build_rows(method, a_metrics, b_metrics, places=places)
    print(f"| Metric | {root_a_name} (mean ± std) | {root_b_name} (mean ± std) | Δ (B−A) |")
    print("|---|---:|---:|---:|")
    for m, a_str, b_str, d_str in rows:
        print(f"| {m} | {a_str} | {b_str} | {d_str} |")


def _print_csv(method: str,
               root_a_name: str,
               root_b_name: str,
               a_metrics: Dict[str, ValAggregate],
               b_metrics: Dict[str, ValAggregate],
               places: int = 4):
    print(f"\n=== {method} ===")
    print(f"metric,{root_a_name}_mean±std,{root_b_name}_mean±std,delta(B-A)")
    rows = _build_rows(method, a_metrics, b_metrics, places=places)
    for m, a_str, b_str, d_str in rows:
        print(f"{m},{a_str},{b_str},{d_str}")


def print_comparison(root_a_name: str,
                     a: Dict[str, Dict[str, ValAggregate]],
                     root_b_name: str,
                     b: Dict[str, Dict[str, ValAggregate]],
                     only_intersection: bool = False,
                     out_format: str = 'table',
                     places: int = 4):
    methods = set(a.keys()) | set(b.keys())
    if only_intersection:
        methods = set(a.keys()) & set(b.keys())
    if not methods:
        print("No methods found in the provided roots.")
        return

    for method in sorted(methods):
        a_metrics = a.get(method) or {}
        b_metrics = b.get(method) or {}
        if not (a_metrics or b_metrics):
            continue
        if out_format == 'markdown':
            _print_markdown(method, root_a_name, root_b_name, a_metrics, b_metrics, places=places)
        elif out_format == 'csv':
            _print_csv(method, root_a_name, root_b_name, a_metrics, b_metrics, places=places)
        else:
            _print_table(method, root_a_name, root_b_name, a_metrics, b_metrics, places=places)


def _build_multi_rows(metrics_list: List[Dict[str, ValAggregate]],
                      places: int = 4) -> List[List[str]]:
    """Build rows for multi-root comparison.

    Returns a list of rows, where each row is [metric, v1, v2, ..., vN].
    """
    all_metrics = set()
    for m in metrics_list:
        all_metrics |= set(m.keys())

    def fmt(v: Optional[ValAggregate]) -> str:
        if not v or v.mean is None:
            return 'NA'
        if v.std is not None:
            return f"{format_float(v.mean, places, keep_trailing=True)} ± {format_float(v.std, places, keep_trailing=True)}"
        return f"{format_float(v.mean, places, keep_trailing=True)}"

    rows: List[List[str]] = []
    for metric in sorted(all_metrics):
        row = [metric]
        for m in metrics_list:
            row.append(fmt(m.get(metric)))
        rows.append(row)
    return rows


def _print_multi_table(method: str,
                       names: List[str],
                       metrics_list: List[Dict[str, ValAggregate]],
                       places: int = 4):
    n_list = [_pick_nfolds(m) for m in metrics_list]
    subtitle_parts = [f"{names[i]}{f' (n={n_list[i]})' if n_list[i] else ''}" for i in range(len(names))]
    print(f"\n=== {method} ===")
    print(' vs '.join(subtitle_parts))

    headers = ['Metric'] + [f"{nm} (mean ± std)" for nm in names]
    rows = _build_multi_rows(metrics_list, places=places)
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(str(x)) for x in col) for col in cols]
    # clamp metric column width to a reasonable size
    if widths:
        widths[0] = min(widths[0], 28)

    def trunc(s: str, w: int) -> str:
        return s if len(s) <= w else (s[: w - 1] + '…')

    def hline(ch: str = '-') -> str:
        return '+' + '+'.join(ch * (w + 2) for w in widths) + '+'

    print(hline('-'))
    print('| ' + ' | '.join(
        trunc(headers[i], widths[i]).ljust(widths[i]) if i == 0 else trunc(headers[i], widths[i]).rjust(widths[i])
        for i in range(len(headers))
    ) + ' |')
    print(hline('='))
    for row in rows:
        formatted = [
            trunc(row[0], widths[0]).ljust(widths[0])
        ] + [
            trunc(val, widths[i + 1]).rjust(widths[i + 1]) for i, val in enumerate(row[1:])
        ]
        print('| ' + ' | '.join(formatted) + ' |')
    print(hline('-'))


def _print_multi_markdown(method: str,
                          names: List[str],
                          metrics_list: List[Dict[str, ValAggregate]],
                          places: int = 4):
    n_list = [_pick_nfolds(m) for m in metrics_list]
    subtitle_parts = [f"{names[i]}{f' (n={n_list[i]})' if n_list[i] else ''}" for i in range(len(names))]
    print(f"\n### {method}")
    print(' vs '.join(subtitle_parts))
    headers = ['Metric'] + [f"{nm} (mean ± std)" for nm in names]
    print('| ' + ' | '.join(headers) + ' |')
    print('|---' + '|---:' * len(names) + '|')
    rows = _build_multi_rows(metrics_list, places=places)
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')


def _print_multi_csv(method: str,
                     names: List[str],
                     metrics_list: List[Dict[str, ValAggregate]],
                     places: int = 4):
    print(f"\n=== {method} ===")
    headers = ['metric'] + [f"{nm}_mean±std" for nm in names]
    print(','.join(headers))
    rows = _build_multi_rows(metrics_list, places=places)
    for row in rows:
        print(','.join(row))


def _latex_escape(s: str) -> str:
    if s is None:
        return ''
    # Minimal escaping for LaTeX special chars in text fields
    repl = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    out = ''.join(repl.get(ch, ch) for ch in str(s))
    return out


def _format_val_for_latex(val: Optional[ValAggregate], places: int, include_std: bool) -> str:
    if not val or val.mean is None:
        return 'NA'
    m = format_float(val.mean, places, keep_trailing=True)
    if include_std and (val.std is not None):
        s = format_float(val.std, places, keep_trailing=True)
        # Render std as a math subscript of the mean
        return f"${m}_{{{s}}}$"
    return m


def _apply_latex_highlight(cell: str, style: str) -> str:
    r"""Apply highlighting to a LaTeX cell, handling math-mode values.

    Only the mean is highlighted:
      - Bold: $m_{s}$ -> $\mathbf{m}_{s}$
      - Underline: $m_{s}$ -> $\underline{m}_{s}$

    Non-math cells use \textbf{...} or \underline{...} on the whole cell.
    """
    is_math = cell.startswith('$') and cell.endswith('$')
    if not is_math:
        if style == 'bold':
            return f"\\textbf{{{cell}}}"
        if style == 'underline':
            return f"\\underline{{{cell}}}"
        return cell

    # Math-mode: split mean and optional subscript
    inner = cell[1:-1]
    mean_part = inner
    sub_part = ''
    # Detect pattern: m_{s}
    if '_{' in inner:
        idx = inner.find('_{')
        mean_part = inner[:idx]
        sub_part = inner[idx:]  # includes leading _{

    if style == 'bold':
        new_inner = f"\\mathbf{{{mean_part}}}{sub_part}"
        return f"${new_inner}$"
    if style == 'underline':
        new_inner = f"\\underline{{{mean_part}}}{sub_part}"
        return f"${new_inner}$"
    return cell


def _best_and_second_indices(values: List[Optional[float]], mode: str = 'max') -> Tuple[Set[int], Set[int]]:
    # values: per-root numeric means (or None) for a fixed method+metric
    idx_vals = [(i, v) for i, v in enumerate(values) if v is not None]
    if not idx_vals:
        return set(), set()
    # Determine ordering direction
    reverse = (mode == 'max')
    # Sort by value with specified direction
    idx_vals.sort(key=lambda t: t[1], reverse=reverse)
    # Collect unique value levels in order
    unique_vals: List[float] = []
    for _, v in idx_vals:
        if not unique_vals or v != unique_vals[-1]:
            unique_vals.append(v)
    if not unique_vals:
        return set(), set()
    top = unique_vals[0]
    second = unique_vals[1] if len(unique_vals) > 1 else None
    best = {i for i, v in idx_vals if v == top}
    second_best = {i for i, v in idx_vals if (second is not None and v == second)}
    return best, second_best


def _print_latex_across_methods(names: List[str],
                                results_list: List[Dict[str, Dict[str, ValAggregate]]],
                                include_std: bool,
                                only_intersection: bool,
                                places: int,
                                outfile: Optional[str],
                                caption: Optional[str],
                                label: Optional[str],
                                win_mode: str = 'max',
                                compile_pdf: bool = False,
                                keep_aux: bool = False,
                                engine: Optional[str] = None,
                                include_precision: bool = False,
                                include_recall: bool = False,
                                include_summary: bool = True,
                                include_param_table: bool = True,
                                include_embeddings: bool = True,
                                include_time: bool = True,
                                linear_flags: Optional[List[bool]] = None) -> None:
    # Determine methods to include, ignoring linear encoders for per-method tables
    if linear_flags is None:
        linear_flags = [False] * len(names)
    # Consider only non-linear encoders when building the comparable method set
    non_linear_indices = [i for i, lf in enumerate(linear_flags) if not lf]
    method_sets = [set(results_list[i].keys()) for i in non_linear_indices if results_list[i]]
    if method_sets:
        methods = sorted(set.intersection(*method_sets) if only_intersection else set.union(*method_sets))
    else:
        methods = []
    if not methods:
        content = '% No methods found to include in LaTeX table.'
        if outfile:
            outp = Path(outfile)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(content + "\n", encoding='utf-8')
        else:
            print(content)
        return

    # Determine metric columns: fixed order baseline + optional precision/recall
    base_metrics: List[str] = [
        'acc', 'balanced_acc', 'f1_macro', 'f1_weighted', 'roc_auc_macro', 'roc_auc_weighted'
    ]
    opt_metrics: List[str] = []
    if include_precision:
        opt_metrics += ['precision_macro', 'precision_weighted']
    if include_recall:
        opt_metrics += ['recall_macro', 'recall_weighted']
    if include_embeddings:
        opt_metrics += ['avg_embeddings_per_wsi']
    if include_time:
        opt_metrics += ['train_time_hours', 'train_time_total_hours']
    metrics = base_metrics + opt_metrics
    # Column spec with vertical rules: |Method|Encoder|metrics...|
    # Adds visual column separators as requested
    colspec = '|' + '|'.join(['l', 'l'] + ['r'] * len(metrics)) + '|'

    lines: List[str] = []
    # Track how many tables have been emitted on the current page
    # to enforce a hard cap of 3 tables per page.
    max_tables_per_page = 2
    tables_on_current_page = 0
    lines.append('% Auto-generated by compare_val_metrics.py (--format=latex)')
    lines.append(r'% Requires: \usepackage{graphicx} (for \resizebox) and \usepackage{multirow}')
    # Build per-method rows, but chunk into multiple tabulars to avoid page overflow
    # Rows per method reflect only encoders that will be displayed (non-linear)
    visible_encoders = [(i, nm) for i, nm in enumerate(names) if not linear_flags[i]]
    rows_per_method = max(1, len(visible_encoders))
    max_rows_per_table = 40  # heuristic number of encoder-rows per page
    methods_per_chunk = max(1, max_rows_per_table // rows_per_method)
    for start_idx in range(0, len(methods), methods_per_chunk):
        chunk = methods[start_idx:start_idx + methods_per_chunk]
        # Start a chunked table
        lines.append('\\begingroup\\setlength{\\tabcolsep}{4pt}')
        lines.append('\\resizebox{\\textwidth}{!}{%')
        lines.append(f'\\begin{{tabular}}{{{colspec}}}')
        header = ['Method', 'Encoder'] + [_latex_escape(mk) for mk in metrics]
        lines.append('\\hline')
        lines.append(' & '.join(header) + ' \\')
        lines.append('\\hline\\hline')

        for method in chunk:
            # Skip if we have no visible encoders (e.g., all linear): no per-method table
            if not visible_encoders:
                break
            # Build per-metric arrays of numeric means across visible (non-linear) roots
            per_metric_values: Dict[str, List[Optional[float]]] = {}
            for mk in metrics:
                vals: List[Optional[float]] = []
                for (i, _nm) in visible_encoders:
                    v = results_list[i].get(method, {}).get(mk)
                    vals.append(float(v.mean) if (v and v.mean is not None) else None)
                per_metric_values[mk] = vals

            # Precompute highlight indices per metric among visible encoders
            highlight: Dict[str, Tuple[Set[int], Set[int]]] = {}
            for mk in metrics:
                m = mk.lower()
                mode_for_mk = 'min' if ('loss' in m) else win_mode
                highlight[mk] = _best_and_second_indices(per_metric_values[mk], mode=mode_for_mk)

            # Emit rows: one per encoder (name), print method once using \multirow
            n_rows = len(visible_encoders)
            for j, (i, nm) in enumerate(visible_encoders):
                row_cells: List[str] = []
                if j == 0:
                    # Print method name once spanning all visible encoders
                    row_cells.append(f"\\multirow{{{n_rows}}}{{*}}{{{_latex_escape(method)}}}")
                else:
                    # Empty first cell (continued multirow)
                    row_cells.append('')
                row_cells.append(_latex_escape(nm))
                for mk in metrics:
                    v = results_list[i].get(method, {}).get(mk)
                    cell = _format_val_for_latex(v, places, include_std)
                    best_set, second_set = highlight[mk]
                    if j in best_set:
                        cell = _apply_latex_highlight(cell, 'bold')
                    elif j in second_set:
                        cell = _apply_latex_highlight(cell, 'underline')
                    row_cells.append(cell)
                lines.append(' & '.join(row_cells) + ' \\')
            # Strong separator between different methods
            lines.append('\\hline\\hline')

        # Close chunk
        if lines and lines[-1] == '\\hline\\hline':
            lines[-1] = '\\hline'
        else:
            lines.append('\\hline')
        lines.append('\\end{tabular}')
        lines.append('}')  # end resizebox
        lines.append('\\endgroup')
        # Enforce at most 3 tables per page
        tables_on_current_page += 1
        if tables_on_current_page >= max_tables_per_page:
            # Use clearpage to force a true page break even in two-column docs
            lines.append('\\clearpage')
            tables_on_current_page = 0

    # Optional summary table: average over methods per encoder
    if include_summary and (methods or any(linear_flags)):
        # Build aggregated values per encoder and metric
        # Compute mean and (optional) std across methods of the per-method means
        agg_vals: List[Dict[str, ValAggregate]] = []  # per-encoder -> metric -> aggregate
        for i in range(len(names)):
            enc_map: Dict[str, ValAggregate] = {}
            for mk in metrics:
                acc: List[float] = []
                # Choose per-encoder method set: for linear encoders, use their own methods
                if linear_flags[i]:
                    methods_i = sorted(results_list[i].keys())
                else:
                    methods_i = methods
                for method in methods_i:
                    vm = results_list[i].get(method, {}).get(mk)
                    if vm and vm.mean is not None:
                        try:
                            acc.append(float(vm.mean))
                        except Exception:
                            pass
                if acc:
                    from math import fsum
                    import statistics
                    mean_v = fsum(acc) / len(acc)
                    std_v = statistics.pstdev(acc) if (include_std and len(acc) > 1) else None
                    enc_map[mk] = ValAggregate(mean=mean_v, std=std_v, n_folds=None)
            agg_vals.append(enc_map)

        # Determine highlight across encoders for each metric
        highlight_sum: Dict[str, Tuple[Set[int], Set[int]]] = {}
        for mk in metrics:
            vals: List[Optional[float]] = []
            for i in range(len(names)):
                v = agg_vals[i].get(mk)
                vals.append(float(v.mean) if (v and v.mean is not None) else None)
            m = mk.lower()
            mode_for_mk = 'min' if ('loss' in m) else win_mode
            highlight_sum[mk] = _best_and_second_indices(vals, mode=mode_for_mk)

        # Emit summary table
        lines.append('')
        lines.append('% Summary: average across methods (per encoder)')
        lines.append('\\begingroup\\setlength{\\tabcolsep}{4pt}')
        lines.append('\\resizebox{\\textwidth}{!}{%')
        lines.append(f'\\begin{{tabular}}{{{colspec}}}')
        header2 = ['Method', 'Encoder'] + [_latex_escape(mk) for mk in metrics]
        lines.append('\\hline')
        lines.append(' & '.join(header2) + ' \\')
        lines.append('\\hline\\hline')
        n_rows = len(names)
        for i, nm in enumerate(names):
            row_cells: List[str] = []
            if i == 0:
                row_cells.append(f"\\multirow{{{n_rows}}}{{*}}{{Average}}")
            else:
                row_cells.append('')
            row_cells.append(_latex_escape(nm))
            for mk in metrics:
                v = agg_vals[i].get(mk)
                cell = _format_val_for_latex(v, places, include_std)
                best_set, second_set = highlight_sum[mk]
                if i in best_set:
                    cell = _apply_latex_highlight(cell, 'bold')
                elif i in second_set:
                    cell = _apply_latex_highlight(cell, 'underline')
                row_cells.append(cell)
            lines.append(' & '.join(row_cells) + ' \\')
        lines.append('\\hline')
        lines.append('\\end{tabular}')
        lines.append('}')  # end resizebox
        lines.append('\\endgroup')
        # Enforce at most 3 tables per page
        tables_on_current_page += 1
        if tables_on_current_page >= max_tables_per_page:
            lines.append('\\clearpage')
            tables_on_current_page = 0

    # Optional parameters table: per-encoder parameter counts
    if include_param_table and names:
        # Known parameter counts (exact integers)
        KNOWN_PARAMS: Dict[str, int] = {
            'resnet50': 23508032,
            'hibou_b': 85741056,
            'phikon_v1': 85798656,
            'uni_v1': 303350784,
            'phikon_v2': 303351808,
            'hibou_l': 303659264,
            'conch_v15': 306109184,
            'conch_v1': 395232769,
            'virchow': 631229184,
            'virchow_v1': 631229184,
            'virchow2': 631239424,
            'virchow_v2': 631239424,
            'uni_v2': 681394176,
            'hoptimus_0': 1134774272,
            'hoptimus_1': 1134774272,
            'gigapath': 1134953984,
            'midnight12k': 1136480768,
        }
        # Synonyms that map to the known keys
        SYNONYMS: Dict[str, str] = {
            'virchow_v1': 'virchow',
            'virchow_v2': 'virchow2',
        }

        def _param_cell_for(name: str) -> str:
            key = SYNONYMS.get(name, name)
            n = KNOWN_PARAMS.get(key)
            if n is None:
                # Fallback for unknown encoders
                return '(~100M)'
            m_str = f"{n/1e6:.2f}M"
            return f"{n} (~{m_str})"

        # Build param table rows in the same order as names
        lines.append('')
        lines.append('% Encoder parameter counts')
        colspec_params = '|' + '|'.join(['l', 'r']) + '|'
        lines.append('\\begingroup\\setlength{\\tabcolsep}{4pt}')
        lines.append('\\resizebox{\\textwidth}{!}{%')
        lines.append(f'\\begin{{tabular}}{{{colspec_params}}}')
        lines.append('\\hline')
        lines.append('Encoder & \\# Params \\')
        lines.append('\\hline\\hline')
        for nm in names:
            enc = _latex_escape(nm)
            cell = _latex_escape(_param_cell_for(nm))
            lines.append(f'{enc} & {cell} \\')
        lines.append('\\hline')
        lines.append('\\end{tabular}')
        lines.append('}')  # end resizebox
        lines.append('\\endgroup')
        # Enforce at most 3 tables per page
        tables_on_current_page += 1
        if tables_on_current_page >= max_tables_per_page:
            lines.append('\\clearpage')
            tables_on_current_page = 0

    # Normalize: ensure every data row inside each tabular ends with a double backslash
    # (header and all data rows), but do not touch rule commands like \hline/\cline.
    try:
        idx = 0
        while True:
            b = next(i for i in range(idx, len(lines)) if lines[i].lstrip().startswith('\\begin{tabular}{'))
            e = next(i for i in range(b + 1, len(lines)) if lines[i].lstrip().startswith('\\end{tabular}'))
            for j in range(b + 1, e):
                ln = lines[j].rstrip()
                stripped = ln.lstrip()
                if not stripped or stripped.startswith('\\hline') or stripped.startswith('\\cline') or stripped.startswith('\\toprule') or stripped.startswith('\\midrule') or stripped.startswith('\\bottomrule') or stripped.startswith('\\cmidrule'):
                    continue
                if ln and not ln.endswith('\\\\'):
                    lines[j] = ln + ' \\\\'
            idx = e + 1
    except StopIteration:
        pass

    content = '\n'.join(lines) + '\n'
    outp: Optional[Path] = None
    if outfile:
        outp = Path(outfile)
        outp.parent.mkdir(parents=True, exist_ok=True)
        # Ensure we always start fresh: remove existing .tex if present
        try:
            outp.unlink(missing_ok=True)
        except TypeError:
            # Python < 3.8 compatibility: emulate missing_ok
            try:
                if outp.exists():
                    outp.unlink()
            except Exception:
                pass
        except Exception:
            pass
        outp.write_text(content, encoding='utf-8')
    else:
        print(content)

    # Optionally compile to PDF
    if compile_pdf and outp is not None:
        _compile_latex_to_pdf(outp, keep_aux=keep_aux, engine=engine)


def _compile_latex_to_pdf(tex_path: Path, keep_aux: bool = False, engine: Optional[str] = None) -> Optional[Path]:
    # Ensure output directory exists
    out_dir = tex_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tex_path.stem
    # Choose engine
    eng = engine
    if eng is None:
        for cand in ['pdflatex', 'xelatex', 'lualatex', 'tectonic']:
            if shutil.which(cand):
                eng = cand
                break
    if eng is None:
        print(f"LaTeX engine not found (pdflatex/xelatex/lualatex). Skipping PDF compile for {tex_path}.")
        return None
    # Create a minimal wrapper that inputs the provided .tex (which contains the table env)
    wrapper_name = f"{stem}__wrapper.tex"
    wrapper_path = out_dir / wrapper_name
    inner_basename = tex_path.name  # keep extension for \input to be explicit
    wrapper = (
        "\\documentclass{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{multirow}\n"
        # Booktabs is optional; we keep grid rules via | and \hline/\cline
        # "\\usepackage{booktabs}\n"
        "\\begin{document}\n"
        f"\\input{{{inner_basename}}}\n"
        "\\end{document}\n"
    )
    wrapper_path.write_text(wrapper, encoding='utf-8')
    # Build command (run in out_dir; pass wrapper by name)
    wrapper_name_only = wrapper_path.name
    if os.path.basename(eng) == 'tectonic':
        # Write outputs to current directory (out_dir)
        cmd = [eng, '-o', '.', wrapper_name_only]
    else:
        cmd = [eng, '-interaction=nonstopmode', '-halt-on-error', '-output-directory', str(out_dir), '-jobname', stem, wrapper_name_only]
    try:
        subprocess.run(cmd, cwd=str(out_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pdf_path = out_dir / f"{stem}.pdf"
        if os.path.basename(eng) == 'tectonic':
            # Tectonic uses wrapper name for the output; rename to stem
            wrapper_pdf = out_dir / f"{wrapper_path.stem}.pdf"
            if wrapper_pdf.exists():
                try:
                    wrapper_pdf.rename(pdf_path)
                except Exception:
                    shutil.copy2(wrapper_pdf, pdf_path)
        if pdf_path.exists():
            print(f"Saved PDF to: {pdf_path}")
        else:
            print(f"Tried to compile LaTeX but PDF not found at: {pdf_path}")
    except subprocess.CalledProcessError as e:
        # Print last lines of output for debugging
        out = e.stdout.decode('utf-8', errors='ignore') if e.stdout else ''
        msg_tail = '\n'.join(out.splitlines()[-30:]) if out else ''
        print(f"LaTeX compile failed for {tex_path}. Engine={eng}. Output tail:\n{msg_tail}")
        # Fallback: try tectonic if available and not already used
        if os.path.basename(eng) != 'tectonic' and shutil.which('tectonic'):
            print("Falling back to 'tectonic' engine...")
            try:
                cmd2 = ['tectonic', '-o', '.', wrapper_name_only]
                subprocess.run(cmd2, cwd=str(out_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                pdf_path = out_dir / f"{stem}.pdf"
                wrapper_pdf = out_dir / f"{wrapper_path.stem}.pdf"
                if wrapper_pdf.exists():
                    try:
                        wrapper_pdf.rename(pdf_path)
                    except Exception:
                        shutil.copy2(wrapper_pdf, pdf_path)
                if pdf_path.exists():
                    print(f"Saved PDF to: {pdf_path}")
                    return pdf_path
            except subprocess.CalledProcessError as e2:
                out2 = e2.stdout.decode('utf-8', errors='ignore') if e2.stdout else ''
                msg_tail2 = '\n'.join(out2.splitlines()[-30:]) if out2 else ''
                print(f"Tectonic compile also failed. Output tail:\n{msg_tail2}")
        return None
    finally:
        # Cleanup wrapper and aux files unless requested to keep
        try:
            wrapper_path.unlink(missing_ok=True)
        except Exception:
            pass
    if not keep_aux:
        exts = ['aux', 'log', 'out', 'toc', 'fls', 'fdb_latexmk']
        for ext in exts:
            for base in [stem, Path(wrapper_name).stem]:
                p = out_dir / f"{base}.{ext}"
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
    return out_dir / f"{stem}.pdf"


def _build_delta_results_vs_baseline(names: List[str],
                                     results_list: List[Dict[str, Dict[str, ValAggregate]]],
                                     baseline_name: str,
                                     only_intersection: bool = False,
                                     compute_std: bool = False) -> Optional[List[Dict[str, Dict[str, ValAggregate]]]]:
    """Build a results_list where values are deltas vs. the baseline encoder.

    For each encoder `e` and each (method, metric), compute:
        delta[e, method, metric] = mean[e, method, metric] - mean[baseline, method, metric]

    Missing values remain absent (rendered as 'NA' later). Methods are included
    consistently across encoders to preserve the same selection behavior as the
    non-delta table (respecting `only_intersection`).
    Returns None if the baseline is not present in `names`.
    """
    if baseline_name not in names:
        return None
    try:
        base_idx = names.index(baseline_name)
    except ValueError:
        return None

    # Determine target method set from the original results, mirroring the
    # selection logic inside _print_latex_across_methods.
    method_sets = [set(r.keys()) for r in results_list if r]
    if method_sets:
        target_methods = sorted(set.intersection(*method_sets) if only_intersection else set.union(*method_sets))
    else:
        target_methods = []

    base_map = results_list[base_idx]
    delta_list: List[Dict[str, Dict[str, ValAggregate]]] = []
    for i, _ in enumerate(names):
        cur_map = results_list[i]
        out_enc: Dict[str, Dict[str, ValAggregate]] = {}
        for method in target_methods:
            # Collect metric keys from both baseline and current encoder for this method
            mkeys: Set[str] = set()
            if method in base_map:
                mkeys |= set(base_map[method].keys())
            if method in cur_map:
                mkeys |= set(cur_map[method].keys())
            # Compute deltas when both sides have a numeric mean
            out_metrics: Dict[str, ValAggregate] = {}
            for mk in mkeys:
                vb = base_map.get(method, {}).get(mk)
                vc = cur_map.get(method, {}).get(mk)
                if vb and vc and (vb.mean is not None) and (vc.mean is not None):
                    try:
                        diff = float(vc.mean) - float(vb.mean)
                        sd: Optional[float] = None
                        if compute_std:
                            sb = float(vb.std) if (vb.std is not None) else None
                            sc = float(vc.std) if (vc.std is not None) else None
                            if (sb is not None) and (sc is not None):
                                from math import sqrt
                                sd = sqrt(sb * sb + sc * sc)
                        out_metrics[mk] = ValAggregate(mean=diff, std=sd, n_folds=None)
                    except Exception:
                        # Skip non-numeric
                        pass
            # Ensure method key exists even if no metrics (so method set is preserved)
            out_enc[method] = out_metrics
        delta_list.append(out_enc)
    return delta_list


def print_multi_comparison(names: List[str],
                           results_list: List[Dict[str, Dict[str, ValAggregate]]],
                           only_intersection: bool = False,
                           out_format: str = 'table',
                           places: int = 4):
    if not results_list:
        print("No results to compare.")
        return
    # Determine methods to show
    method_sets = [set(r.keys()) for r in results_list]
    if only_intersection and method_sets:
        methods = set.intersection(*method_sets)
    elif method_sets:
        methods = set.union(*method_sets)
    else:
        methods = set()
    if not methods:
        print("No methods found in the provided roots.")
        return

    for method in sorted(methods):
        metrics_list = [r.get(method) or {} for r in results_list]
        if not any(metrics_list):
            continue
        if out_format == 'markdown':
            _print_multi_markdown(method, names, metrics_list, places=places)
        elif out_format == 'csv':
            _print_multi_csv(method, names, metrics_list, places=places)
        else:
            _print_multi_table(method, names, metrics_list, places=places)


def _compute_win_tally(a: Dict[str, Dict[str, ValAggregate]],
                       b: Dict[str, Dict[str, ValAggregate]],
                       metric: str,
                       mode: str = 'max',
                       only_intersection: bool = True) -> Tuple[int, int, int, int, List[str], List[str], List[str]]:
    """Compute A vs B wins for the given metric and return details.

    - Considers only methods present in both `a` and `b` when `only_intersection=True`.
    - `metric` may include a split prefix like "val/f1_weighted"; only the final key is used.
    - `mode` either 'max' (higher is better) or 'min' (lower is better).

    Returns:
        (a_wins, b_wins, ties, compared_methods,
         a_winner_methods, b_winner_methods, tie_methods)
    """
    key = metric.split('/', 1)[-1] if isinstance(metric, str) else str(metric)
    methods = set(a.keys()) | set(b.keys())
    if only_intersection:
        methods = set(a.keys()) & set(b.keys())

    a_wins = b_wins = ties = compared = 0
    a_winner_methods: List[str] = []
    b_winner_methods: List[str] = []
    tie_methods: List[str] = []
    for m in sorted(methods):
        am = a.get(m, {}).get(key)
        bm = b.get(m, {}).get(key)
        if not (am and bm):
            continue
        if am.mean is None or bm.mean is None:
            continue
        compared += 1
        va = float(am.mean)
        vb = float(bm.mean)
        if mode == 'min':
            va, vb = -va, -vb  # invert to reuse the same comparison
        if va > vb:
            a_wins += 1
            a_winner_methods.append(m)
        elif vb > va:
            b_wins += 1
            b_winner_methods.append(m)
        else:
            ties += 1
            tie_methods.append(m)
    return a_wins, b_wins, ties, compared, a_winner_methods, b_winner_methods, tie_methods


def main():
    p = argparse.ArgumentParser(description="Compare validation metrics averaged over folds between result roots.")
    # Backward-compatible two-root positional args, plus optional extra roots
    p.add_argument('root_a', type=str, nargs='?', help='First root directory containing per-method subfolders with fold_summary.json')
    p.add_argument('root_b', type=str, nargs='?', help='Second root directory containing per-method subfolders with fold_summary.json')
    p.add_argument('more_roots', type=str, nargs='*', help='Additional root directories for multi-way comparison')

    # Names: legacy for two roots, or multiple via --names
    p.add_argument('--name-a', type=str, default=None, help='Short label for first root (e.g., uni_v1)')
    p.add_argument('--name-b', type=str, default=None, help='Short label for second root (e.g., slide_hubert)')
    p.add_argument('--names', type=str, nargs='+', help='Names for multiple roots, in the same order as provided roots')

    p.add_argument('--only-intersection', action='store_true', help='Show only methods present in all roots (or both in two-root mode)')
    p.add_argument('--format', choices=['table', 'markdown', 'csv', 'latex'], default='table', help='Output format for readability')
    # LaTeX options
    p.add_argument('--latex-file', type=str, default=None, help='If set and --format=latex, write LaTeX table to this file')
    p.add_argument('--latex-include-std', action='store_true', help="Include '± std' in LaTeX cell values")
    p.add_argument('--latex-caption', type=str, default=None, help='Optional LaTeX caption for the table')
    p.add_argument('--latex-label', type=str, default=None, help='Optional LaTeX label for the table (e.g., tab:comparison)')
    # LaTeX metric column selection
    p.add_argument('--latex-include-precision', action='store_true', help='Include precision_macro and precision_weighted columns')
    p.add_argument('--latex-include-recall', action='store_true', help='Include recall_macro and recall_weighted columns')
    # Default ON: embeddings column included unless explicitly disabled
    p.add_argument('--latex-include-embeddings', action='store_true', default=True, help='Include avg_embeddings_per_wsi column (default: on)')
    p.add_argument('--latex-no-embeddings', action='store_true', help='Disable avg_embeddings_per_wsi column')
    # Default ON: training time columns included unless explicitly disabled
    p.add_argument('--latex-include-time', action='store_true', default=True, help='Include training time columns (default: on)')
    p.add_argument('--latex-no-time', action='store_true', help='Disable training time columns')
    p.add_argument('--latex-engine', type=str, choices=['auto', 'pdflatex', 'xelatex', 'lualatex', 'tectonic'], default='auto', help='Engine to compile LaTeX to PDF')
    p.add_argument('--latex-no-summary', action='store_true', help='Do not append an average-across-methods summary table')
    p.add_argument('--places', type=int, default=3, help='Decimal places for numbers')
    # Win tally controls (two-root mode only)
    p.add_argument('--win-metric', type=str, default='val/f1_weighted', help="Metric used for win tally, e.g., 'val/f1_weighted'")
    p.add_argument('--win-mode', choices=['max', 'min'], default='max', help="Optimization direction for win tally: 'max' (higher is better) or 'min' (lower is better)")
    # Plotting controls
    p.add_argument('--save-plots', action='store_true', help='Save histogram/bar plots comparing selected metrics across roots')
    p.add_argument('--plots-dir', type=str, default=None, help='Directory to save plots (default: logs/compare_plots if logs exists, else ./plots)')
    # LaTeX compile controls
    p.add_argument('--latex-compile', action='store_true', help='Also compile the .tex to PDF in the output dir')
    p.add_argument('--latex-keep-aux', action='store_true', help='Keep LaTeX aux/log files when compiling')
    # Extra LaTeX table controls
    p.add_argument('--latex-no-params', action='store_true', help='Do not append the encoder parameter-count table')
    # Delta table controls
    p.add_argument('--include-std-delta', action='store_true', help='Include "+/- std" in delta tables vs backbone (if available)')
    # Exclude methods option
    p.add_argument(
        '--exclude-methods',
        type=str,
        nargs='*',
        default=None,
        help=(
            'Method folder names to exclude from comparison. '
            'Accepts space-separated list (e.g., --exclude-methods clam transmil rrt) '
            'or a single comma-separated string (e.g., --exclude-methods clam,transmil,rrt).'
        ),
    )
    args = p.parse_args()

    # Aggregate roots from positional arguments
    roots: List[str] = []
    if args.root_a:
        roots.append(args.root_a)
    if args.root_b:
        roots.append(args.root_b)
    if args.more_roots:
        roots.extend(args.more_roots)

    if len(roots) == 0:
        p.error('Please provide at least one result root directory.')

    # Determine names
    names: List[str]
    if args.names:
        if len(args.names) != len(roots):
            p.error(f"--names expects {len(roots)} values, got {len(args.names)}")
        names = list(args.names)
    elif len(roots) == 2 and (args.name_a or args.name_b):
        names = [args.name_a or 'A', args.name_b or 'B']
    else:
        # Default to base directory names
        names = [os.path.basename(os.path.normpath(r)) or r for r in roots]

    # Normalize exclude list
    exclude_set: Optional[Set[str]] = None
    if args.exclude_methods:
        # Support both space-separated and single comma-separated usage
        items: List[str] = []
        for it in args.exclude_methods:
            if ',' in it:
                items.extend([x.strip() for x in it.split(',') if x.strip()])
            else:
                if it.strip():
                    items.append(it.strip())
        exclude_set = {s.lower() for s in items}

    # Helper to filter a results dict by excluded method names
    def _filter_results(results: Dict[str, Dict[str, ValAggregate]]) -> Dict[str, Dict[str, ValAggregate]]:
        if not exclude_set:
            return results
        return {k: v for k, v in results.items() if k.lower() not in exclude_set}

    # Single-root: just print per-method metrics
    if len(roots) == 1:
        r0 = collect_root_results(roots[0])
        r0 = _filter_results(r0)
        if not r0:
            print("No results found. Ensure each method folder contains a fold_summary.json.")
            return
        # Reuse multi printer with a single column
        if args.format == 'latex':
            include_embeddings = (False if args.latex_no_embeddings else args.latex_include_embeddings)
            include_time = (False if args.latex_no_time else args.latex_include_time)
            _print_latex_across_methods(
                names,
                [r0],
                include_std=args.latex_include_std,
                only_intersection=args.only_intersection,
                places=args.places,
                outfile=args.latex_file,
                caption=args.latex_caption,
                label=args.latex_label,
                win_mode=args.win_mode,
                compile_pdf=args.latex_compile,
                keep_aux=args.latex_keep_aux,
                engine=(None if args.latex_engine == 'auto' else args.latex_engine),
                include_precision=args.latex_include_precision,
                include_recall=args.latex_include_recall,
                include_summary=(not args.latex_no_summary),
                include_param_table=(not args.latex_no_params),
                include_embeddings=include_embeddings,
                include_time=include_time,
                linear_flags=[('linear' in roots[0].lower())],
            )
            # If an encoder named 'backbone' exists, also emit a delta-vs-backbone PDF
            if ('backbone' in names) and args.latex_file:
                delta_results = _build_delta_results_vs_baseline(
                    names, [r0], 'backbone',
                    only_intersection=args.only_intersection,
                    compute_std=args.include_std_delta,
                )
                if delta_results is not None:
                    base = Path(args.latex_file)
                    delta_out = base.with_name(f"{base.stem}__delta_vs_backbone{base.suffix}")
                    delta_caption = (f"{args.latex_caption} (Δ vs backbone)" if args.latex_caption else None)
                    delta_label = (f"{args.latex_label}-delta" if args.latex_label else None)
                    include_embeddings = (False if args.latex_no_embeddings else args.latex_include_embeddings)
                    include_time = (False if args.latex_no_time else args.latex_include_time)
                    _print_latex_across_methods(
                        names,
                        delta_results,
                        include_std=args.include_std_delta,
                        only_intersection=args.only_intersection,
                        places=args.places,
                        outfile=str(delta_out),
                        caption=delta_caption,
                        label=delta_label,
                        win_mode=args.win_mode,
                        compile_pdf=args.latex_compile,
                        keep_aux=args.latex_keep_aux,
                        engine=(None if args.latex_engine == 'auto' else args.latex_engine),
                        include_precision=args.latex_include_precision,
                        include_recall=args.latex_include_recall,
                        include_summary=(not args.latex_no_summary),
                        include_param_table=(not args.latex_no_params),
                        include_embeddings=include_embeddings,
                        include_time=include_time,
                        linear_flags=[('linear' in roots[0].lower())],
                    )
        else:
            print_multi_comparison(names, [r0], only_intersection=args.only_intersection, out_format=args.format, places=args.places)
        return

    # Helper for plotting across roots
    def _save_metric_plots_across_roots(names: List[str],
                                        results_list: List[Dict[str, Dict[str, ValAggregate]]],
                                        only_intersection: bool,
                                        out_dir: Path):
        import math
        import matplotlib
        matplotlib.use('Agg')  # ensure non-interactive backend
        import matplotlib.pyplot as plt

        # Determine comparable set of per-method keys across non-linear roots only
        lin_flags = [('linear' in rt.lower()) for rt in roots]
        non_linear_indices = [i for i, lf in enumerate(lin_flags) if not lf]
        method_sets = [set(results_list[i].keys()) for i in non_linear_indices if results_list[i]]
        if method_sets:
            methods = set.intersection(*method_sets) if only_intersection else set.union(*method_sets)
        else:
            methods = set()

        if not methods:
            # Nothing to aggregate; still try per-root metrics if they exist directly
            return

        # Helper to aggregate a metric per root by averaging over available methods
        def aggregate_per_root(metric_key: str) -> List[Optional[float]]:
            vals: List[Optional[float]] = []
            for i, r in enumerate(results_list):
                acc: List[float] = []
                # For linear encoders, aggregate over whatever methods they have
                if lin_flags[i]:
                    methods_i = list(r.keys())
                else:
                    methods_i = methods
                for m in methods_i:
                    vm = r.get(m, {}).get(metric_key)
                    if vm and vm.mean is not None:
                        try:
                            acc.append(float(vm.mean))
                        except Exception:
                            pass
                vals.append((sum(acc) / len(acc)) if len(acc) > 0 else None)
            return vals

        # Define metric groups
        weighted_metrics = [
            'f1_weighted', 'balanced_acc', 'precision_weighted', 'recall_weighted', 'roc_auc_weighted'
        ]
        macro_metrics = [
            'f1_macro', 'acc', 'precision_macro', 'recall_macro', 'roc_auc_macro'
        ]

        def plot_group(metrics: List[str], title: str, filename: str):
            # Build figure with one subplot per metric
            n = len(metrics)
            fig_h = 2.2 * n
            fig, axes = plt.subplots(n, 1, figsize=(max(6.0, 1.2 * len(names)), fig_h), squeeze=False)
            axes = axes.ravel()
            made_any = False
            for i, mk in enumerate(metrics):
                ax = axes[i]
                y = aggregate_per_root(mk)
                # Skip metric if no values at all
                if not any(v is not None for v in y):
                    ax.set_visible(False)
                    continue
                made_any = True
                x = list(range(len(names)))
                vals = [v if (v is not None) else float('nan') for v in y]
                bars = ax.bar(x, vals, color='#4C72B0')
                ax.set_xticks(x)
                ax.set_xticklabels(names, rotation=15, ha='right')
                # Dynamic y-limits to highlight small differences (clamped to [0,1])
                finite_vals = [v for v in vals if not math.isnan(v)]
                if finite_vals:
                    vmin, vmax = min(finite_vals), max(finite_vals)
                    span = max(vmax - vmin, 1e-3)  # avoid zero span
                    margin = 0.15 * span
                    lo = max(0.0, vmin - margin)
                    hi = min(1.0, vmax + margin)
                    # Ensure a minimum visible range
                    if hi - lo < 0.02:
                        mid = 0.5 * (lo + hi)
                        lo = max(0.0, mid - 0.01)
                        hi = min(1.0, mid + 0.01)
                    ax.set_ylim(lo, hi)
                else:
                    ax.set_ylim(0.0, 1.0)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_ylabel(mk)
                # Annotate bars
                for b, v in zip(bars, vals):
                    if not (v is None or math.isnan(v)):
                        ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
            if not made_any:
                plt.close(fig)
                return
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            out_path = out_dir / filename
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

        # Ensure out directory exists
        out_dir.mkdir(parents=True, exist_ok=True)
        # 1) Global comparisons averaged over methods
        plot_group(weighted_metrics, 'Comparison: Weighted metrics (avg over methods)', 'comparison_weighted.png')
        plot_group(macro_metrics, 'Comparison: Macro metrics (avg over methods)', 'comparison_macro.png')

        # 2) Per-method plots: one figure per method showing the same metric groups
        def _sanitize(name: str) -> str:
            safe = ''.join(ch if ch.isalnum() or ch in ('_', '-', '.') else '_' for ch in name)
            return safe.strip('._') or 'method'

        methods_dir = out_dir / 'methods'
        methods_dir.mkdir(parents=True, exist_ok=True)

        for method in sorted(methods):
            # Gather values per root for each metric
            def plot_method_group(metrics: List[str], title: str, filename: str):
                n = len(metrics)
                fig_h = 2.2 * n
                fig, axes = plt.subplots(n, 1, figsize=(max(6.0, 1.2 * len(names)), fig_h), squeeze=False)
                axes = axes.ravel()
                any_plotted = False
                for i, mk in enumerate(metrics):
                    ax = axes[i]
                    vals = []
                    for r in results_list:
                        v = r.get(method, {}).get(mk)
                        vals.append(float(v.mean) if (v and v.mean is not None) else float('nan'))
                    if not any(not math.isnan(v) for v in vals):
                        ax.set_visible(False)
                        continue
                    any_plotted = True
                    x = list(range(len(names)))
                    bars = ax.bar(x, vals, color='#4C72B0')
                    ax.set_xticks(x)
                    ax.set_xticklabels(names, rotation=15, ha='right')
                    # Dynamic y-limits per method metric
                    finite_vals = [v for v in vals if not math.isnan(v)]
                    if finite_vals:
                        vmin, vmax = min(finite_vals), max(finite_vals)
                        span = max(vmax - vmin, 1e-3)
                        margin = 0.15 * span
                        lo = max(0.0, vmin - margin)
                        hi = min(1.0, vmax + margin)
                        if hi - lo < 0.02:
                            mid = 0.5 * (lo + hi)
                            lo = max(0.0, mid - 0.01)
                            hi = min(1.0, mid + 0.01)
                        ax.set_ylim(lo, hi)
                    else:
                        ax.set_ylim(0.0, 1.0)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    ax.set_ylabel(mk)
                    for b, v in zip(bars, vals):
                        if not math.isnan(v):
                            ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
                if not any_plotted:
                    plt.close(fig)
                    return
                fig.suptitle(title)
                fig.tight_layout(rect=[0, 0, 1, 0.98])
                fig.savefig(filename, dpi=200)
                plt.close(fig)

            sm = _sanitize(method)
            plot_method_group(
                weighted_metrics,
                f"{method}: Weighted metrics across roots",
                methods_dir / f"{sm}_weighted.png",
            )
            plot_method_group(
                macro_metrics,
                f"{method}: Macro metrics across roots",
                methods_dir / f"{sm}_macro.png",
            )

    # Two-root: preserve legacy output with delta and win tally
    if len(roots) == 2:
        a, b = compare_roots(roots[0], roots[1])
        a = _filter_results(a)
        b = _filter_results(b)

        if not a and not b:
            print("No comparable results found. Ensure each method folder contains a fold_summary.json.")
            return

        if args.format == 'latex':
            include_embeddings = (False if args.latex_no_embeddings else args.latex_include_embeddings)
            include_time = (False if args.latex_no_time else args.latex_include_time)
            _print_latex_across_methods(
                names,
                [a, b],
                include_std=args.latex_include_std,
                only_intersection=args.only_intersection,
                places=args.places,
                outfile=args.latex_file,
                caption=args.latex_caption,
                label=args.latex_label,
                win_mode=args.win_mode,
                compile_pdf=args.latex_compile,
                keep_aux=args.latex_keep_aux,
                engine=(None if args.latex_engine == 'auto' else args.latex_engine),
                include_precision=args.latex_include_precision,
                include_recall=args.latex_include_recall,
                include_summary=(not args.latex_no_summary),
                include_param_table=(not args.latex_no_params),
                include_embeddings=include_embeddings,
                include_time=include_time,
                linear_flags=[('linear' in roots[0].lower()), ('linear' in roots[1].lower())],
            )
            # Also emit a delta PDF if 'backbone' is among the encoder names
            if ('backbone' in names) and args.latex_file:
                delta_results = _build_delta_results_vs_baseline(
                    names, [a, b], 'backbone',
                    only_intersection=args.only_intersection,
                    compute_std=args.include_std_delta,
                )
                if delta_results is not None:
                    base = Path(args.latex_file)
                    delta_out = base.with_name(f"{base.stem}__delta_vs_backbone{base.suffix}")
                    delta_caption = (f"{args.latex_caption} (Δ vs backbone)" if args.latex_caption else None)
                    delta_label = (f"{args.latex_label}-delta" if args.latex_label else None)
                    include_embeddings = (False if args.latex_no_embeddings else args.latex_include_embeddings)
                    include_time = (False if args.latex_no_time else args.latex_include_time)
                    _print_latex_across_methods(
                        names,
                        delta_results,
                        include_std=args.include_std_delta,
                        only_intersection=args.only_intersection,
                        places=args.places,
                        outfile=str(delta_out),
                        caption=delta_caption,
                        label=delta_label,
                        win_mode=args.win_mode,
                        compile_pdf=args.latex_compile,
                        keep_aux=args.latex_keep_aux,
                        engine=(None if args.latex_engine == 'auto' else args.latex_engine),
                        include_precision=args.latex_include_precision,
                        include_recall=args.latex_include_recall,
                        include_summary=(not args.latex_no_summary),
                        include_param_table=(not args.latex_no_params),
                        include_embeddings=include_embeddings,
                        include_time=include_time,
                        linear_flags=[('linear' in roots[0].lower()), ('linear' in roots[1].lower())],
                    )
        else:
            print_comparison(
                names[0],
                a,
                names[1],
                b,
                only_intersection=args.only_intersection,
                out_format=args.format,
                places=args.places,
            )
        # Print win tally summary
        a_w, b_w, ties, compared, a_win_methods, b_win_methods, tie_methods = _compute_win_tally(
            a, b, metric=args.win_metric, mode=args.win_mode, only_intersection=args.only_intersection
        )
        direction = 'higher' if args.win_mode == 'max' else 'lower'
        print(f"\nWin summary by '{args.win_metric}' ({direction} is better) on {compared} comparable methods:")
        print(f"  {names[0]}: {a_w} | {names[1]}: {b_w} | ties: {ties}")
        # Also list which methods each side won (and ties) for transparency
        if a_win_methods:
            print(f"  {names[0]} wins: {', '.join(a_win_methods)}")
        else:
            print(f"  {names[0]} wins: -")
        if b_win_methods:
            print(f"  {names[1]} wins: {', '.join(b_win_methods)}")
        else:
            print(f"  {names[1]} wins: -")
        if tie_methods:
            print(f"  Ties: {', '.join(tie_methods)}")
        # Save plots if requested
        if args.save_plots:
            # Determine output directory
            default_dir = Path('logs/compare_plots') if Path('logs').exists() else Path('plots')
            plots_dir = Path(args.plots_dir) if args.plots_dir else default_dir
            _save_metric_plots_across_roots(names, [a, b], args.only_intersection, plots_dir)
            print(f"Saved plots to: {plots_dir}")
        return

    # Multi-root (>=3): print multi-way comparison table/markdown/csv
    results_list = collect_roots_multi(roots)
    # Apply filtering to each root's results
    results_list = [_filter_results(r) for r in results_list]
    if not any(results_list):
        print("No comparable results found. Ensure each method folder contains a fold_summary.json.")
        return
    if args.format == 'latex':
        include_embeddings = (False if args.latex_no_embeddings else args.latex_include_embeddings)
        include_time = (False if args.latex_no_time else args.latex_include_time)
        _print_latex_across_methods(
            names,
            results_list,
            include_std=args.latex_include_std,
            only_intersection=args.only_intersection,
            places=args.places,
            outfile=args.latex_file,
            caption=args.latex_caption,
            label=args.latex_label,
            win_mode=args.win_mode,
            compile_pdf=args.latex_compile,
            keep_aux=args.latex_keep_aux,
            engine=(None if args.latex_engine == 'auto' else args.latex_engine),
            include_precision=args.latex_include_precision,
            include_recall=args.latex_include_recall,
            include_summary=(not args.latex_no_summary),
            include_param_table=(not args.latex_no_params),
            include_embeddings=include_embeddings,
            include_time=include_time,
            linear_flags=[('linear' in r.lower()) for r in roots],
        )
        # Also emit a delta PDF if 'backbone' is among the encoder names
        if ('backbone' in names) and args.latex_file:
            delta_results = _build_delta_results_vs_baseline(
                names, results_list, 'backbone',
                only_intersection=args.only_intersection,
                compute_std=args.include_std_delta,
            )
            if delta_results is not None:
                base = Path(args.latex_file)
                delta_out = base.with_name(f"{base.stem}__delta_vs_backbone{base.suffix}")
                delta_caption = (f"{args.latex_caption} (Δ vs backbone)" if args.latex_caption else None)
                delta_label = (f"{args.latex_label}-delta" if args.latex_label else None)
                include_embeddings = (False if args.latex_no_embeddings else args.latex_include_embeddings)
                include_time = (False if args.latex_no_time else args.latex_include_time)
                _print_latex_across_methods(
                    names,
                    delta_results,
                    include_std=args.include_std_delta,
                    only_intersection=args.only_intersection,
                    places=args.places,
                    outfile=str(delta_out),
                    caption=delta_caption,
                    label=delta_label,
                    win_mode=args.win_mode,
                    compile_pdf=args.latex_compile,
                    keep_aux=args.latex_keep_aux,
                    engine=(None if args.latex_engine == 'auto' else args.latex_engine),
                    include_precision=args.latex_include_precision,
                    include_recall=args.latex_include_recall,
                    include_summary=(not args.latex_no_summary),
                    include_param_table=(not args.latex_no_params),
                    include_embeddings=include_embeddings,
                    include_time=include_time,
                    linear_flags=[('linear' in r.lower()) for r in roots],
                )
    else:
        print_multi_comparison(
            names,
            results_list,
            only_intersection=args.only_intersection,
            out_format=args.format,
            places=args.places,
        )
    # Save plots if requested for multi-root case
    if args.save_plots:
        default_dir = Path('logs/compare_plots') if Path('logs').exists() else Path('plots')
        plots_dir = Path(args.plots_dir) if args.plots_dir else default_dir
        _save_metric_plots_across_roots(names, results_list, args.only_intersection, plots_dir)
        print(f"Saved plots to: {plots_dir}")
    return
    """
    Examples:
      Two roots (legacy):
        python scripts/compare_val_metrics.py /path/to/uni_v1/CAMELYON17 /path/to/slide_hubert/CAMELYON17 --name-a uni_v1 --name-b slide_hubert

      Multiple roots (3+):
        python scripts/compare_val_metrics.py \
            /home/mila/k/kotpy/scratch/MIL-Lab/results/slide_hubert/CAMELYON17 \
            /home/mila/k/kotpy/scratch/MIL-Lab/results/slide_hubert_clusters_50/CAMELYON17 \
            /home/mila/k/kotpy/scratch/MIL-Lab/results/slide_hubert_50_100/CAMELYON17 \
            /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v1/CAMELYON17 \
            --names slide_hubert clusters_50 hubert_50_100 uni_v1

    """


if __name__ == '__main__':
    main()
