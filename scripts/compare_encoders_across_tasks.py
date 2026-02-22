#!/usr/bin/env python3
"""
Compare encoder performance across tasks by averaging over methods.

For each (dataset, task) combination, this script:
1. Computes the average performance of each encoder across all methods
2. Determines which encoder performs best
3. Aggregates wins across all tasks to show overall encoder comparison

Directory structure expected:
    results/<encoder_name>/<dataset_name>/<task_name>/<method_name>/fold_summary.json
"""
import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


@dataclass
class ValAggregate:
    mean: float
    std: Optional[float] = None
    n_folds: Optional[int] = None


METRIC_DISPLAY_NAMES = {
    'f1_weighted': 'F1 (weighted)',
    'roc_auc_weighted': 'ROC-AUC (weighted)',
    'balanced_acc': 'Balanced Acc',
    'f1_macro': 'F1 (macro)',
    'roc_auc_macro': 'ROC-AUC (macro)',
    'acc': 'Accuracy',
    'precision_weighted': 'Precision (weighted)',
    'recall_weighted': 'Recall (weighted)',
    'precision_macro': 'Precision (macro)',
    'recall_macro': 'Recall (macro)',
}

DATASET_DISPLAY_NAMES = {
    'TCGA': 'TCGA',
    'TCGA_BRCA': 'TCGA-BRCA',
    'TCGA_COAD': 'TCGA-COAD',
    'TCGA_ESCA': 'TCGA-ESCA',
    'TCGA_SARC': 'TCGA-SARC',
    'TCGA_TGCT': 'TCGA-TGCT',
    'TCGA_THYM': 'TCGA-THYM',
    'TCGA_UCEC': 'TCGA-UCEC',
    'bc_therapy': 'BC Therapy',
    'bracs': 'BRACS',
    'camelyon17': 'Camelyon17',
    'cptac_all': 'CPTAC',
    'cptac_brca': 'CPTAC-BRCA',
    'cptac_ccrcc': 'CPTAC-CCRCC',
    'cptac_coad': 'CPTAC-COAD',
    'cptac_gbm': 'CPTAC-GBM',
    'cptac_hnsc': 'CPTAC-HNSC',
    'cptac_lscc': 'CPTAC-LSCC',
    'cptac_luad': 'CPTAC-LUAD',
    'cptac_lung': 'CPTAC-Lung',
    'cptac_ov': 'CPTAC-OV',
    'cptac_pda': 'CPTAC-PDA',
    'cptac_ucec': 'CPTAC-UCEC',
    'dhmc_kidney': 'DHMC-Kidney',
    'dhmc_luad': 'DHMC-LUAD',
    'ebrains': 'eBrains',
    'imp': 'IMP',
    'imp_cervix': 'IMP-Cervix',
    'mbc': 'MBC',
    'mut-het-rcc': 'Mut-Het-RCC',
    'nadt': 'NADT',
    'natbrca': 'NatBRCA',
    'panda': 'PANDA',
}

TASK_DISPLAY_NAMES = {
    'cancer_type_classification': 'Cancer Type',
    'primary_diagnosis': 'Primary Diagnosis',
    'er_status': 'ER Status',
    'grade': 'Grade',
    'her2_status': 'HER2 Status',
    'residual_cancer_burden': 'Residual Cancer Burden',
    'coarse': 'Coarse Grading',
    'fine': 'Fine Grading',
    'breast_cancer_metastases': 'Metastases',
    'organ': 'Organ',
    'subtype': 'Subtype',
    'morphological_subtyping': 'Morphological Subtype',
    'histologic_pattern': 'Histologic Pattern',
    'diagnosis': 'Diagnosis',
    'diagnosis_group': 'Diagnosis Group',
    'idh_status': 'IDH Status',
    'dysplasia_grading': 'Dysplasia Grading',
    'treatment_response': 'Treatment Response',
    'response': 'Response',
    'lymphovascular_invasion': 'Lymphovascular Invasion',
    'prostate_cancer_grade': 'Prostate Cancer Grade',
    # Immune
    'Immune_class': 'Immune Class',
    # Mutations
    'APC_mutation': 'APC Mut.',
    'ACVR2A_mutation': 'ACVR2A Mut.',
    'ARID1A_mutation': 'ARID1A Mut.',
    'BAP1_mutation': 'BAP1 Mut.',
    'CASP8_mutation': 'CASP8 Mut.',
    'CTNNB1_mutation': 'CTNNB1 Mut.',
    'EGFR_mutation': 'EGFR Mut.',
    'KEAP1_mutation': 'KEAP1 Mut.',
    'KRAS_mutation': 'KRAS Mut.',
    'PBRM1_mutation': 'PBRM1 Mut.',
    'PIK3CA_mutation': 'PIK3CA Mut.',
    'PTEN_mutation': 'PTEN Mut.',
    'SETD1B_mutation': 'SETD1B Mut.',
    'SETD2_mutation': 'SETD2 Mut.',
    'SMAD4_mutation': 'SMAD4 Mut.',
    'STK11_mutation': 'STK11 Mut.',
    'TP53_mutation': 'TP53 Mut.',
    'VHL_mutation': 'VHL Mut.',
    # Grades / histology
    'Histologic_Grade': 'Histologic Grade',
    'MSI_H': 'MSI-H',
}


def _is_number(x) -> bool:
    try:
        return isinstance(x, (int, float)) and not (isinstance(x, float) and (x != x))
    except Exception:
        return False


def _find_fold_summary_file(method_dir: str) -> Optional[str]:
    """Find fold_summary.json inside a method directory."""
    candidate = os.path.join(method_dir, 'fold_summary.json')
    if os.path.isfile(candidate):
        return candidate
    for root, _, files in os.walk(method_dir):
        if 'fold_summary.json' in files:
            return os.path.join(root, 'fold_summary.json')
    return None


def _load_val_aggregate(summary_path: str) -> Dict[str, ValAggregate]:
    """Load validation aggregate metrics from fold_summary.json."""
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
                    out[k] = ValAggregate(
                        mean=float(mean),
                        std=float(std) if _is_number(std) else None,
                        n_folds=int(n) if isinstance(n, int) else None
                    )
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

    import statistics
    out: Dict[str, ValAggregate] = {}
    for k, vs in values.items():
        if not vs:
            continue
        mean = sum(vs) / len(vs)
        std = statistics.pstdev(vs) if len(vs) > 1 else 0.0
        out[k] = ValAggregate(mean=mean, std=std, n_folds=len(vs))

    return out


def scan_results_directory(results_root: str) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, ValAggregate]]]]]:
    """
    Scan results directory and return structure:
    {encoder: {dataset: {task: {method: {metric: ValAggregate}}}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    results_path = Path(results_root)

    if not results_path.exists():
        print(f"Results directory does not exist: {results_root}")
        return {}

    # Iterate over encoders
    for encoder_dir in results_path.iterdir():
        if not encoder_dir.is_dir() or encoder_dir.name.startswith('.'):
            continue
        encoder_name = encoder_dir.name

        # Iterate over datasets
        for dataset_dir in encoder_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                continue
            dataset_name = dataset_dir.name

            # Iterate over tasks
            for task_dir in dataset_dir.iterdir():
                if not task_dir.is_dir() or task_dir.name.startswith('.'):
                    continue
                task_name = task_dir.name

                # Iterate over methods
                for method_dir in task_dir.iterdir():
                    if not method_dir.is_dir() or method_dir.name.startswith('.'):
                        continue
                    method_name = method_dir.name

                    # Find and load fold_summary.json
                    summary_path = _find_fold_summary_file(str(method_dir))
                    if summary_path:
                        try:
                            metrics = _load_val_aggregate(summary_path)
                            results[encoder_name][dataset_name][task_name][method_name] = metrics
                        except Exception as e:
                            print(f"Warning: Failed to load {summary_path}: {e}")

    return dict(results)


def compute_encoder_avg_per_task(
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, ValAggregate]]]]],
    metric: str,
    method_filter: Optional[List[str]] = None
) -> Dict[Tuple[str, str], Dict[str, Tuple[float, int]]]:
    """
    Compute average performance across methods for each encoder per (dataset, task).

    Args:
        results: Nested dict of results
        metric: Metric name to average
        method_filter: Optional list of method names to include (None = all methods)

    Returns: {(dataset, task): {encoder: (avg_metric_value, num_methods)}}
    """
    task_encoder_avgs = defaultdict(dict)
    method_set = set(method_filter) if method_filter else None

    for encoder_name, datasets in results.items():
        for dataset_name, tasks in datasets.items():
            for task_name, methods in tasks.items():
                # Collect metric values across filtered methods for this encoder
                values = []
                for method_name, metrics in methods.items():
                    # Skip if method filter is active and this method is not included
                    if method_set is not None and method_name not in method_set:
                        continue

                    val_agg = metrics.get(metric)
                    if val_agg and val_agg.mean is not None:
                        values.append(val_agg.mean)

                # Compute average
                if values:
                    avg = sum(values) / len(values)
                    task_encoder_avgs[(dataset_name, task_name)][encoder_name] = (avg, len(values))

    return dict(task_encoder_avgs)


def find_non_applicable_encoders(
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, ValAggregate]]]]],
    tasks: List[Tuple[str, str]],
    method_filter: Optional[List[str]] = None
) -> Dict[Tuple[str, str], List[str]]:
    """
    Identify encoders that have zero methods available for each task (after filtering).
    """
    method_set = set(method_filter) if method_filter else None
    non_applicable = {}

    for dataset, task in tasks:
        missing: List[str] = []
        for encoder_name, encoder_data in results.items():
            task_methods = encoder_data.get(dataset, {}).get(task, {})
            if not task_methods:
                missing.append(encoder_name)
                continue

            filtered_methods = [
                m for m in task_methods.keys()
                if method_set is None or m in method_set
            ]
            if not filtered_methods:
                missing.append(encoder_name)

        if missing:
            non_applicable[(dataset, task)] = sorted(missing)

    return non_applicable


def determine_winners(
    task_encoder_avgs: Dict[Tuple[str, str], Dict[str, Tuple[float, int]]],
    mode: str = 'max'
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], str]]:
    """
    Determine which encoder wins in each task.

    Returns:
        - wins_count: {encoder: number_of_wins}
        - task_winners: {(dataset, task): winning_encoder}
    """
    wins_count = defaultdict(int)
    task_winners = {}

    for (dataset, task), encoder_avgs in task_encoder_avgs.items():
        if not encoder_avgs:
            continue

        # Find the best encoder (by avg value, first element of tuple)
        if mode == 'max':
            best_encoder = max(encoder_avgs.items(), key=lambda x: x[1][0])
        else:  # mode == 'min'
            best_encoder = min(encoder_avgs.items(), key=lambda x: x[1][0])

        winner_name, (winner_value, _) = best_encoder
        wins_count[winner_name] += 1
        task_winners[(dataset, task)] = winner_name

    return dict(wins_count), task_winners

def plot_encoder_comparison(
    task_encoder_avgs: Dict[Tuple[str, str], Dict[str, Tuple[float, int]]],
    task_winners: Dict[Tuple[str, str], str],
    metric: str,
    mode: str,
    output_path: str,
    methods_used: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 8),
    dpi: int = 300
):
    """Create a grouped bar plot comparing encoders across all tasks."""

    # Sort tasks for consistent ordering
    sorted_tasks = sorted(task_encoder_avgs.keys())

    # Get all unique encoders across all tasks
    all_encoders_unsorted = set(
        encoder
        for encoder_avgs in task_encoder_avgs.values()
        for encoder in encoder_avgs.keys()
    )

    # Calculate wins for each encoder
    wins_count = defaultdict(int)
    for winner in task_winners.values():
        wins_count[winner] += 1
    total_tasks = len(task_winners)

    # Sort encoders by number of wins (descending), then alphabetically
    all_encoders = sorted(all_encoders_unsorted,
                         key=lambda x: (-wins_count.get(x, 0), x))

    # Prepare data for plotting
    n_tasks = len(sorted_tasks)
    n_encoders = len(all_encoders)

    # Dynamically adjust figure width based on number of tasks
    # Allocate more width per task for better visibility
    width_per_task = 0.8  # inches per task (reduced for smaller file size)
    dynamic_width = max(figsize[0], n_tasks * width_per_task)
    dynamic_figsize = (dynamic_width, figsize[1])

    print(f"Creating plot with {n_tasks} tasks and {n_encoders} encoders")
    print(f"Figure size: {dynamic_figsize[0]:.1f} x {dynamic_figsize[1]:.1f} inches at {dpi} DPI")
    print(f"Output resolution: {int(dynamic_figsize[0] * dpi)} x {int(dynamic_figsize[1] * dpi)} pixels")

    # Create color map for encoders
    colors = plt.cm.Set2(np.linspace(0, 1, n_encoders))
    encoder_colors = {encoder: colors[i] for i, encoder in enumerate(all_encoders)}

    # Set up the plot
    fig, ax = plt.subplots(figsize=dynamic_figsize, dpi=dpi)

    # Set bar width and positions
    # Make bars much wider for better visibility
    bar_width = 1.6 / n_encoders  # Wide bars
    x = np.arange(n_tasks) * 2.2  # Spacing to accommodate bars

    # Plot bars for each encoder
    for i, encoder in enumerate(all_encoders):
        values = []
        for task in sorted_tasks:
            encoder_avgs = task_encoder_avgs[task]
            if encoder in encoder_avgs:
                values.append(encoder_avgs[encoder][0])  # [0] is the avg value
            else:
                values.append(0)  # Missing data

        offset = (i - n_encoders/2 + 0.5) * bar_width
        # Add win count to label
        wins = wins_count.get(encoder, 0)
        label = f"{encoder} ({wins}/{total_tasks})"
        bars = ax.bar(x + offset, values, bar_width, label=label,
                     color=encoder_colors[encoder], alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add star markers for winners
        for j, task in enumerate(sorted_tasks):
            if task_winners.get(task) == encoder and values[j] > 0:
                ax.plot(x[j] + offset, values[j], marker='*', markersize=18,
                       color='gold', markeredgecolor='black', markeredgewidth=1.0, zorder=10)

    # Customize plot
    ax.set_xlabel('Task', fontsize=24, fontweight='bold')
    ax.set_ylabel(f'{metric}', fontsize=24, fontweight='bold')

    mode_text = 'higher is better' if mode == 'max' else 'lower is better'
    title = f'Encoder Comparison Across Tasks\n{metric} ({mode_text})'
    # if methods_used:
    #     title += f"\nMethods averaged: {', '.join(sorted(methods_used))}"
    # else:
    #     title += "\nMethods averaged: all available"
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)

    # Set x-axis labels with color coding by dataset
    task_labels = [f"{dataset}/{task}" for dataset, task in sorted_tasks]
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=90, ha='right', fontsize=16, fontweight='bold')

    # Increase y-axis tick label size
    ax.tick_params(axis='y', labelsize=16)

    # Color-code x-axis labels by dataset
    unique_datasets = sorted(set(dataset for dataset, _ in sorted_tasks))
    n_datasets = len(unique_datasets)

    # Use a colormap that can handle many datasets without repeating
    if n_datasets <= 10:
        cmap = plt.cm.tab10
    elif n_datasets <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_rainbow

    # Spread out color indices to maximize contrast between consecutive datasets in sorted order
    # Find the order datasets appear in sorted_tasks
    dataset_order = []
    seen = set()
    for dataset, _ in sorted_tasks:
        if dataset not in seen:
            dataset_order.append(dataset)
            seen.add(dataset)

    # Assign colors in a spread-out pattern (e.g., 0, n/2, n/4, 3n/4, ...)
    color_indices = []
    step = max(1, n_datasets // 2)
    idx = 0
    for _ in range(n_datasets):
        while idx in color_indices:
            idx = (idx + 1) % n_datasets
        color_indices.append(idx)
        idx = (idx + step) % n_datasets

    dataset_color_dict = {}
    for i, dataset in enumerate(dataset_order):
        color_idx = color_indices[i]
        dataset_color_dict[dataset] = cmap(color_idx / max(1, n_datasets - 1))

    # Apply colors to tick labels
    for i, (dataset, task) in enumerate(sorted_tasks):
        ax.get_xticklabels()[i].set_color(dataset_color_dict[dataset])
        ax.get_xticklabels()[i].set_fontweight('bold')

    # Add vertical separators between different datasets
    prev_dataset = None
    for i, (dataset, task) in enumerate(sorted_tasks):
        if prev_dataset is not None and dataset != prev_dataset:
            # Add a vertical line between datasets
            ax.axvline(x=x[i] - 1.1, color='gray', linestyle='--', linewidth=2.5, alpha=0.5, zorder=0)
        prev_dataset = dataset

    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.0)
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=18,
             title='Encoders', title_fontsize=20, framealpha=0.9)

    # Add note about stars and color coding
    notes = '★ = Winner for this task\nX-axis labels colored by dataset'
    fig.text(0.99, 0.01, notes,
            ha='right', va='bottom', fontsize=16, style='italic', fontweight='bold')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from file extension
    file_ext = output_path_obj.suffix.lower()

    # For vector formats (PDF, SVG), don't specify DPI in savefig (already set in figure)
    # For raster formats (PNG, JPG), use the DPI parameter
    if file_ext in ['.pdf', '.svg']:
        plt.savefig(output_path, bbox_inches='tight', format=file_ext[1:])
        print(f"\nPlot saved to: {output_path} (vector format - infinitely zoomable, smaller file size)")
    else:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    plt.close()


def plot_polar_encoder_comparison(
    task_encoder_avgs: Dict[Tuple[str, str], Dict[str, Tuple[float, int]]],
    task_winners: Dict[Tuple[str, str], str],
    metric: str,
    mode: str,
    output_path: str,
    methods_used: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 12),
    dpi: int = 300,
    fill_alpha: float = 0.15,
    line_width: float = 2.5,
    marker_size: int = 8,
    font_scale: float = 1.0,
    zoom_axis: bool = True,
    zoom_padding: float = 0.1
):
    """
    Create a publication-quality polar (radar) plot comparing encoders across tasks.

    Suitable for top-tier AI research papers and conferences.

    Args:
        task_encoder_avgs: {(dataset, task): {encoder: (avg_value, num_methods)}}
        task_winners: {(dataset, task): winning_encoder}
        metric: Name of the metric being compared
        mode: 'max' or 'min' optimization direction
        output_path: Path to save the plot
        methods_used: Optional list of methods that were averaged
        figsize: Figure size in inches
        dpi: Resolution for raster formats
        fill_alpha: Transparency of filled polygons
        line_width: Width of polygon edges
        marker_size: Size of data point markers
        font_scale: Scale factor for all fonts
        zoom_axis: If True, zoom radial axis to data range for better visibility
        zoom_padding: Padding factor for zoomed axis (fraction of data range)
    """
    # Sort tasks for consistent ordering
    sorted_tasks = sorted(task_encoder_avgs.keys())
    n_tasks = len(sorted_tasks)

    if n_tasks < 3:
        print(f"Warning: Polar plot requires at least 3 tasks, found {n_tasks}. Skipping polar plot.")
        return

    # Get all unique encoders and sort by wins
    all_encoders_set = set(
        encoder
        for encoder_avgs in task_encoder_avgs.values()
        for encoder in encoder_avgs.keys()
    )

    wins_count = defaultdict(int)
    for winner in task_winners.values():
        wins_count[winner] += 1

    # Sort encoders by wins (descending), then alphabetically
    all_encoders = sorted(all_encoders_set, key=lambda x: (-wins_count.get(x, 0), x))
    n_encoders = len(all_encoders)

    print(f"Creating polar plot with {n_tasks} tasks and {n_encoders} encoders")

    # Extract values and handle missing data
    values = {encoder: [] for encoder in all_encoders}
    for task in sorted_tasks:
        encoder_avgs = task_encoder_avgs[task]
        for encoder in all_encoders:
            if encoder in encoder_avgs:
                values[encoder].append(encoder_avgs[encoder][0])
            else:
                values[encoder].append(np.nan)

    # Calculate data range for axis zooming
    all_values = [v for enc_vals in values.values() for v in enc_vals if not np.isnan(v)]
    data_min = min(all_values)
    data_max = max(all_values)
    data_range = data_max - data_min

    if zoom_axis and data_range > 0:
        # Zoom into data range with padding
        padding = data_range * zoom_padding
        axis_min = max(0, data_min - padding)  # Don't go below 0 for metrics like accuracy
        axis_max = data_max + padding
        print(f"Zooming radial axis to [{axis_min:.3f}, {axis_max:.3f}] (data range: {data_min:.3f}-{data_max:.3f})")
    else:
        axis_min = 0
        axis_max = data_max * 1.1 if data_max > 0 else 1.0

    # Set up the polar plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'), dpi=dpi)

    # Calculate angles for each task
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Professional color palette suitable for publications
    # Using colorblind-friendly colors
    publication_colors = [
        '#E64B35',  # Vermillion red
        '#4DBBD5',  # Cyan
        '#00A087',  # Teal
        '#3C5488',  # Navy blue
        '#F39B7F',  # Salmon
        '#8491B4',  # Steel blue
        '#91D1C2',  # Mint
        '#DC0000',  # Red
        '#7E6148',  # Brown
        '#B09C85',  # Tan
    ]

    # Extend colors if needed
    if n_encoders > len(publication_colors):
        extra_colors = plt.cm.Set3(np.linspace(0, 1, n_encoders - len(publication_colors)))
        publication_colors.extend([plt.matplotlib.colors.rgb2hex(c) for c in extra_colors])

    # Different line styles for accessibility (colorblind-friendly)
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

    # Plot each encoder
    for i, encoder in enumerate(all_encoders):
        encoder_values = values[encoder]

        # Close the polygon
        encoder_values_closed = encoder_values + [encoder_values[0]]

        # Handle NaN values for plotting
        valid_mask = [not np.isnan(v) for v in encoder_values_closed]

        color = publication_colors[i % len(publication_colors)]
        linestyle = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        # Create label with win count
        wins = wins_count.get(encoder, 0)
        total_tasks = len(task_winners)
        label = f"{encoder} ({wins}/{total_tasks} wins)"

        # Plot the line (only valid points)
        if all(valid_mask):
            ax.plot(angles, encoder_values_closed,
                   color=color, linewidth=line_width * font_scale,
                   linestyle=linestyle, label=label, zorder=3)
            ax.fill(angles, encoder_values_closed,
                   color=color, alpha=fill_alpha, zorder=2)

            # Add markers at data points
            ax.scatter(angles[:-1], encoder_values,
                      color=color, s=marker_size * font_scale ** 2,
                      marker=marker, zorder=4, edgecolors='white', linewidths=0.5)
        else:
            # Plot segments between valid points
            valid_angles = [a for a, v in zip(angles[:-1], encoder_values) if not np.isnan(v)]
            valid_vals = [v for v in encoder_values if not np.isnan(v)]
            if valid_angles:
                ax.scatter(valid_angles, valid_vals,
                          color=color, s=marker_size * font_scale ** 2,
                          marker=marker, zorder=4, label=label,
                          edgecolors='white', linewidths=0.5)

    # Create task labels (shortened for readability)
    task_labels = []
    for dataset, task in sorted_tasks:
        # Shorten dataset name if too long
        short_dataset = dataset[:12] + '...' if len(dataset) > 15 else dataset
        short_task = task[:10] + '...' if len(task) > 12 else task
        task_labels.append(f"{short_dataset}\n{short_task}")

    # Set the task labels at each angle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(task_labels, fontsize=int(9 * font_scale), fontweight='medium')

    # Set radial axis limits (zoomed to data range)
    ax.set_ylim(axis_min, axis_max)

    # Create evenly spaced ticks within the zoomed range
    n_ticks = 5
    tick_values = np.linspace(axis_min, axis_max, n_ticks)
    ax.set_yticks(tick_values)
    ax.set_yticklabels([f'{y:.2f}' for y in tick_values],
                      fontsize=int(8 * font_scale), color='gray')

    # Style the grid
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_linewidth(1.5)
    ax.spines['polar'].set_color('gray')

    # Add title
    mode_text = 'higher is better' if mode == 'max' else 'lower is better'
    title = f'Encoder Performance Comparison\n{metric} ({mode_text})'
    ax.set_title(title, fontsize=int(14 * font_scale), fontweight='bold',
                pad=20, y=1.08)

    # Add legend outside the plot
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.15, 1.0),
        fontsize=int(10 * font_scale),
        title='Encoders',
        title_fontsize=int(11 * font_scale),
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=False
    )
    legend.get_title().set_fontweight('bold')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    file_ext = output_path_obj.suffix.lower()

    if file_ext in ['.pdf', '.svg']:
        plt.savefig(output_path, bbox_inches='tight', format=file_ext[1:],
                   facecolor='white', edgecolor='none')
        print(f"Polar plot saved to: {output_path} (vector format)")
    else:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Polar plot saved to: {output_path}")

    plt.close()


def print_results(
    task_encoder_avgs: Dict[Tuple[str, str], Dict[str, Tuple[float, int]]],
    wins_count: Dict[str, int],
    task_winners: Dict[Tuple[str, str], str],
    metric: str,
    mode: str,
    places: int = 4,
    methods_used: Optional[List[str]] = None,
    non_applicable: Optional[Dict[Tuple[str, str], List[str]]] = None
):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"Encoder Comparison Summary")
    print(f"Metric: {metric} ({'higher is better' if mode == 'max' else 'lower is better'})")
    if methods_used:
        print(f"Methods averaged: {', '.join(sorted(methods_used))}")
    else:
        print(f"Methods averaged: all available")
    print(f"{'='*80}\n")

    # Print per-task results
    print("Per-Task Results:")
    print("-" * 80)

    for (dataset, task), encoder_avgs in sorted(task_encoder_avgs.items()):
        print(f"\n{dataset} / {task}:")
        winner = task_winners.get((dataset, task))

        # Sort encoders by performance
        sorted_encoders = sorted(
            encoder_avgs.items(),
            key=lambda x: x[1][0],
            reverse=(mode == 'max')
        )

        for encoder_name, (avg_value, num_methods) in sorted_encoders:
            marker = " ★" if encoder_name == winner else ""
            print(f"  {encoder_name:30s}: {avg_value:.{places}f} (avg over {num_methods} methods){marker}")

        if non_applicable:
            missing_encoders = non_applicable.get((dataset, task), [])
            if missing_encoders:
                print(f"  Non-applicable (no methods found): {', '.join(missing_encoders)}")
                print(f"  Task excluded from wins due to missing encoders.")

    # Print overall wins summary
    print(f"\n{'='*80}")
    print("Overall Wins Summary:")
    print("-" * 80)

    total_tasks = len(task_winners)
    skipped_tasks = len(non_applicable) if non_applicable else 0
    sorted_wins = sorted(wins_count.items(), key=lambda x: x[1], reverse=True)

    for encoder_name, num_wins in sorted_wins:
        percentage = (num_wins / total_tasks * 100) if total_tasks > 0 else 0
        print(f"  {encoder_name:30s}: {num_wins:3d} / {total_tasks} ({percentage:5.1f}%)")

    print(f"Comparable tasks counted: {total_tasks}")
    if skipped_tasks:
        print(f"Skipped (missing methods for at least one encoder): {skipped_tasks}")

    print(f"{'='*80}\n")

    # Print non-applicable summary
    if non_applicable:
        print("Non-applicable Tasks (no methods found for these encoders):")
        print("-" * 80)
        for (dataset, task), encoders in sorted(non_applicable.items()):
            print(f"  {dataset}/{task}: {', '.join(encoders)}")
        print(f"{'='*80}\n")


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    for char in ['_', '&', '%', '#', '$']:
        text = text.replace(char, f'\\{char}')
    return text


def _format_metric_entry(
    mean: float, std: Optional[float], places: int, rank: int
) -> str:
    """
    Format a metric entry as LaTeX math.

    rank=1 -> bold (best), rank=2 -> underline (second best), else plain.
    Standard deviation is shown as a subscript.
    """
    mean_str = f'{mean:.{places}f}'

    if std is not None:
        std_str = f'{std:.{places}f}'
        if rank == 1:
            return f'$\\mathbf{{{mean_str}}}_{{\\pm{std_str}}}$'
        elif rank == 2:
            return f'$\\underline{{{mean_str}}}_{{\\pm{std_str}}}$'
        return f'${mean_str}_{{\\pm{std_str}}}$'

    if rank == 1:
        return f'$\\mathbf{{{mean_str}}}$'
    elif rank == 2:
        return f'$\\underline{{{mean_str}}}$'
    return f'${mean_str}$'


def collect_metric_data_with_std(
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, ValAggregate]]]]],
    metrics: List[str],
    method_filter: Optional[List[str]] = None,
) -> Dict[Tuple[str, str], Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]]:
    """
    Collect per-task, per-metric, per-encoder (mean, std) data.

    When multiple methods are specified, averages the means and stds across methods.

    Returns:
        {(dataset, task): {metric: {encoder: (mean, std)}}}
    """
    method_set = set(method_filter) if method_filter else None
    task_data: Dict[
        Tuple[str, str],
        Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]],
    ] = {}

    for encoder_name, datasets in results.items():
        for dataset_name, tasks in datasets.items():
            for task_name, methods in tasks.items():
                key = (dataset_name, task_name)
                if key not in task_data:
                    task_data[key] = {m: {} for m in metrics}

                for metric in metrics:
                    means: List[float] = []
                    stds: List[float] = []
                    for method_name, method_metrics in methods.items():
                        if method_set is not None and method_name not in method_set:
                            continue
                        val_agg = method_metrics.get(metric)
                        if val_agg and val_agg.mean is not None:
                            means.append(val_agg.mean)
                            if val_agg.std is not None:
                                stds.append(val_agg.std)

                    if means:
                        avg_mean = sum(means) / len(means)
                        avg_std = sum(stds) / len(stds) if stds else None
                        task_data[key][metric][encoder_name] = (avg_mean, avg_std)

    return task_data


def generate_latex_table(
    metric_data: Dict[
        Tuple[str, str],
        Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]],
    ],
    encoders: List[str],
    metrics: List[str],
    mode: str = 'max',
    places: int = 3,
    methods_used: Optional[List[str]] = None,
    caption: Optional[str] = None,
    label: str = 'tab:encoder_comparison',
) -> str:
    """
    Generate a standalone LaTeX document with a table comparing encoders.

    Rows are grouped by task, with one sub-row per metric.
    Best value per (task, metric) is bold; second-best is underlined.
    """
    n_encoders = len(encoders)
    n_metrics = len(metrics)
    sorted_tasks = sorted(metric_data.keys())

    encoder_headers = ' & '.join(_escape_latex(e) for e in encoders)

    if caption is None:
        method_str = ', '.join(methods_used) if methods_used else 'all'
        caption = (
            f'Encoder comparison across tasks '
            f'(methods: {_escape_latex(method_str)})'
        )

    encoder_col = r'>{\centering\arraybackslash}X'
    col_spec = 'cc' + encoder_col * n_encoders
    lines = [
        r'\documentclass{article}',
        r'\usepackage[landscape,margin=1cm]{geometry}',
        r'\usepackage{booktabs}',
        r'\usepackage{xltabular}',
        r'\usepackage{amsmath}',
        r'\pagestyle{empty}',
        '',
        r'\begin{document}',
        f'\\begin{{xltabular}}{{\\textwidth}}{{{col_spec}}}',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}} \\\\',
        r'\toprule',
        f'Task & Metric & {encoder_headers} \\\\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        f'Task & Metric & {encoder_headers} \\\\',
        r'\midrule',
        r'\endhead',
        r'\bottomrule',
        r'\endfoot',
    ]

    for task_idx, (dataset, task) in enumerate(sorted_tasks):
        task_metrics = metric_data.get((dataset, task), {})
        dataset_display = DATASET_DISPLAY_NAMES.get(dataset, _escape_latex(dataset))
        task_display = TASK_DISPLAY_NAMES.get(task, _escape_latex(task))
        task_label = f'{_escape_latex(dataset_display)}/{_escape_latex(task_display)}'

        for metric_idx, metric in enumerate(metrics):
            encoder_values = task_metrics.get(metric, {})

            # Rank encoders for this task-metric combination
            ranked = [
                (enc, encoder_values[enc][0])
                for enc in encoders
                if enc in encoder_values and encoder_values[enc][0] is not None
            ]
            ranked.sort(key=lambda x: x[1], reverse=(mode == 'max'))
            ranks = {enc: idx + 1 for idx, (enc, _) in enumerate(ranked)}

            # Format each encoder's entry
            entries = []
            for enc in encoders:
                if enc in encoder_values and encoder_values[enc][0] is not None:
                    mean, std = encoder_values[enc]
                    rank = ranks.get(enc, 99)
                    entries.append(
                        _format_metric_entry(mean, std, places, rank)
                    )
                else:
                    entries.append('--')

            entries_str = ' & '.join(entries)
            metric_display = METRIC_DISPLAY_NAMES.get(
                metric, _escape_latex(metric)
            )

            if metric_idx == 0:
                lines.append(
                    f'{task_label} & {metric_display} & {entries_str} \\\\'
                )
            else:
                lines.append(f' & {metric_display} & {entries_str} \\\\')

        if task_idx < len(sorted_tasks) - 1:
            lines.append(r'\midrule')

    lines.extend([
        r'\end{xltabular}',
        r'\end{document}',
    ])

    return '\n'.join(lines)


def compile_latex_to_pdf(tex_path: str) -> Optional[str]:
    """Compile a .tex file to PDF using pdflatex."""
    import subprocess

    tex_path_obj = Path(tex_path)
    output_dir = tex_path_obj.parent

    try:
        pdflatex_cmd = [
            'pdflatex',
            '-interaction=nonstopmode',
            '-output-directory',
            str(output_dir),
            str(tex_path),
        ]
        # Run twice: longtable/xltabular needs two passes to
        # compute correct column widths via the .aux file.
        for _ in range(2):
            subprocess.run(
                pdflatex_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

        pdf_path = tex_path_obj.with_suffix('.pdf')
        if pdf_path.exists():
            for ext in ['.aux', '.log']:
                aux_file = tex_path_obj.with_suffix(ext)
                if aux_file.exists():
                    aux_file.unlink()
            return str(pdf_path)

        log_path = tex_path_obj.with_suffix('.log')
        print(f"Warning: pdflatex failed. Check {log_path}")
        return None
    except FileNotFoundError:
        print(
            "Warning: pdflatex not found. "
            "Install texlive to compile LaTeX to PDF."
        )
        return None
    except subprocess.TimeoutExpired:
        print("Warning: pdflatex timed out after 60 seconds.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare encoder performance across tasks by averaging over methods"
    )
    parser.add_argument(
        '--results_root',
        type=str,
        default='results',
        help='Root directory containing results (default: results)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='f1_weighted',
        help='Metric to compare (default: f1_weighted)'
    )
    parser.add_argument(
        '--mode',
        choices=['max', 'min'],
        default='max',
        help="Optimization direction: 'max' (higher is better) or 'min' (lower is better)"
    )
    parser.add_argument(
        '--places',
        type=int,
        default=3,
        help='Decimal places for output and LaTeX table (default: 3)'
    )
    parser.add_argument(
        '--save-json',
        type=str,
        default=None,
        help='Save detailed results to JSON file'
    )
    parser.add_argument(
        '--encoders',
        type=str,
        nargs='+',
        default=None,
        help='Specific encoder names to include (default: all available encoders). '
             'Example: --encoders iBOT uni_v1 backbone'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=None,
        help='Specific method names to average over (default: all available methods). '
             'Example: --methods abmil clam transmil'
    )
    parser.add_argument(
        '--exclude-tasks',
        type=str,
        default=None,
        help='Path to file containing dataset/task pairs to exclude (one per line, format: dataset/task)'
    )
    parser.add_argument(
        '--plot',
        type=str,
        default="compare.pdf",
        help='Save bar plot to specified file path. Use .pdf or .svg for vector format (smaller, zoomable), or .png for raster format (default: compare.pdf)'
    )
    parser.add_argument(
        '--plot-width',
        type=int,
        default=40,
        help='Plot width in inches (default: 40)'
    )
    parser.add_argument(
        '--plot-height',
        type=int,
        default=12,
        help='Plot height in inches (default: 12)'
    )
    parser.add_argument(
        '--plot-dpi',
        type=int,
        default=300,
        help='Plot DPI (default: 300)'
    )
    parser.add_argument(
        '--polar-plot',
        type=str,
        default="polar.pdf",
        help='Save polar (radar) plot to specified file path. Ideal for publication figures. '
             'Use .pdf or .svg for vector format (default: None, disabled)'
    )
    parser.add_argument(
        '--polar-size',
        type=int,
        default=12,
        help='Polar plot size in inches (square figure, default: 12)'
    )
    parser.add_argument(
        '--polar-fill-alpha',
        type=float,
        default=0.15,
        help='Transparency of filled regions in polar plot (default: 0.15)'
    )
    parser.add_argument(
        '--polar-font-scale',
        type=float,
        default=1.0,
        help='Font scale factor for polar plot (default: 1.0)'
    )
    parser.add_argument(
        '--polar-no-zoom',
        action='store_true',
        help='Disable axis zooming (start radial axis from 0 instead of data minimum)'
    )
    parser.add_argument(
        '--polar-zoom-padding',
        type=float,
        default=0.1,
        help='Padding factor for zoomed axis as fraction of data range (default: 0.1)'
    )
    parser.add_argument(
        '--filter-winner',
        type=str,
        default=None,
        help='Only plot tasks where this encoder wins. If not specified, plot all tasks. '
             'Example: --filter-winner uni_v1'
    )
    parser.add_argument(
        '--include-tasks',
        type=str,
        default=None,
        help='Path to file containing dataset/task pairs to include (one per line, '
             'format: dataset/task). Only these tasks will be shown. '
             'Lines starting with # are treated as comments.'
    )
    parser.add_argument(
        '--latex',
        type=str,
        default='comparison_table.tex',
        help='Generate a LaTeX comparison table and save to this path '
             '(default: comparison_table.tex). A PDF is also produced if '
             'pdflatex is available. Use --no-latex to disable.'
    )
    parser.add_argument(
        '--no-latex',
        action='store_true',
        default=False,
        help='Disable LaTeX table generation'
    )
    parser.add_argument(
        '--table-metrics',
        type=str,
        nargs='+',
        default=['f1_weighted', 'roc_auc_weighted', 'balanced_acc'],
        help='Metrics to include in the LaTeX table '
             '(default: f1_weighted roc_auc_weighted balanced_acc)'
    )

    args = parser.parse_args()

    # Scan results directory
    print(f"Scanning results directory: {args.results_root}")
    if args.encoders:
        print(f"Filtering to encoders: {', '.join(args.encoders)}")
    else:
        print("Including all available encoders")
    if args.methods:
        print(f"Averaging over methods: {', '.join(args.methods)}")
    else:
        print("Averaging over all available methods")
    results = scan_results_directory(args.results_root)

    if not results:
        print("No results found!")
        return

    # Filter encoders if specified
    if args.encoders:
        encoder_set = set(args.encoders)
        available_encoders = set(results.keys())

        # Check if requested encoders exist
        missing = encoder_set - available_encoders
        if missing:
            print(f"Warning: Requested encoders not found: {', '.join(sorted(missing))}")

        # Filter to only requested encoders
        results = {enc: data for enc, data in results.items() if enc in encoder_set}

        if not results:
            print(f"No results found for requested encoders: {', '.join(sorted(args.encoders))}")
            print(f"Available encoders: {', '.join(sorted(available_encoders))}")
            return

        print(f"Comparing encoders: {', '.join(sorted(results.keys()))}\n")
    encoders_being_compared = set(results.keys())

    # Count what was found
    num_encoders = len(results)
    num_datasets = len(set(
        dataset
        for encoder_data in results.values()
        for dataset in encoder_data.keys()
    ))
    num_tasks = len(set(
        (dataset, task)
        for encoder_data in results.values()
        for dataset, dataset_data in encoder_data.items()
        for task in dataset_data.keys()
    ))
    num_methods = sum(
        len(methods)
        for encoder_data in results.values()
        for dataset_data in encoder_data.values()
        for task_data in dataset_data.values()
        for methods in task_data.values()
    )

    print(f"Found: {num_encoders} encoders, {num_datasets} datasets, {num_tasks} tasks, {num_methods} method results\n")

    # Validate methods filter if specified
    if args.methods:
        # Collect all unique method names across all results
        all_methods = set()
        for encoder_data in results.values():
            for dataset_data in encoder_data.values():
                for task_data in dataset_data.values():
                    all_methods.update(task_data.keys())

        requested_methods = set(args.methods)
        missing_methods = requested_methods - all_methods
        if missing_methods:
            print(f"Warning: Requested methods not found: {', '.join(sorted(missing_methods))}")

        available_requested = requested_methods & all_methods
        if not available_requested:
            print(f"No results found for requested methods: {', '.join(sorted(args.methods))}")
            print(f"Available methods: {', '.join(sorted(all_methods))}")
            return

    # Compute encoder averages per task
    task_encoder_avgs = compute_encoder_avg_per_task(results, args.metric, args.methods)

    if not task_encoder_avgs:
        print(f"No tasks found with metric '{args.metric}'")
        return

    # Load and apply task inclusions if specified
    included_tasks = None
    if args.include_tasks:
        included_tasks = set()
        try:
            with open(args.include_tasks, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('/')
                        if len(parts) == 2:
                            included_tasks.add((parts[0], parts[1]))
                        else:
                            print(f"Warning: Invalid format in inclusion file: {line}")

            if included_tasks:
                print(f"\nIncluding only {len(included_tasks)} specified task(s):")
                for dataset, task in sorted(included_tasks):
                    print(f"  + {dataset}/{task}")
                print()

                original_count = len(task_encoder_avgs)
                task_encoder_avgs = {
                    (dataset, task): avgs
                    for (dataset, task), avgs in task_encoder_avgs.items()
                    if (dataset, task) in included_tasks
                }
                filtered_count = original_count - len(task_encoder_avgs)
                print(f"Filtered to {len(task_encoder_avgs)} matching task(s) "
                      f"({filtered_count} excluded)\n")

        except FileNotFoundError:
            print(f"Warning: Inclusion file not found: {args.include_tasks}")
        except Exception as e:
            print(f"Warning: Error reading inclusion file: {e}")

    if not task_encoder_avgs:
        print("No tasks remaining after inclusion filter")
        return

    # Load and apply task exclusions if specified
    excluded_tasks = set()
    if args.exclude_tasks:
        try:
            with open(args.exclude_tasks, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('/')
                        if len(parts) == 2:
                            excluded_tasks.add((parts[0], parts[1]))
                        else:
                            print(f"Warning: Invalid format in exclusion file: {line}")

            if excluded_tasks:
                print(f"\nExcluding {len(excluded_tasks)} task(s) from comparison:")
                for dataset, task in sorted(excluded_tasks):
                    print(f"  - {dataset}/{task}")
                print()

                # Filter out excluded tasks
                original_count = len(task_encoder_avgs)
                task_encoder_avgs = {
                    (dataset, task): avgs
                    for (dataset, task), avgs in task_encoder_avgs.items()
                    if (dataset, task) not in excluded_tasks
                }
                excluded_count = original_count - len(task_encoder_avgs)
                print(f"Filtered out {excluded_count} excluded task(s), {len(task_encoder_avgs)} tasks remaining\n")

        except FileNotFoundError:
            print(f"Warning: Exclusion file not found: {args.exclude_tasks}")
        except Exception as e:
            print(f"Warning: Error reading exclusion file: {e}")

    if not task_encoder_avgs:
        print(f"No tasks remaining after exclusions")
        return

    # Track which encoders have no methods for each task
    non_applicable = find_non_applicable_encoders(
        results,
        list(task_encoder_avgs.keys()),
        args.methods
    )

    # Keep only tasks where every encoder being compared has data; others are excluded from wins
    comparable_task_encoder_avgs = {
        task: avgs
        for task, avgs in task_encoder_avgs.items()
        if set(avgs.keys()) == encoders_being_compared
    }

    # Determine winners
    wins_count, task_winners = determine_winners(comparable_task_encoder_avgs, args.mode)

    # Filter to only tasks where specified encoder wins (for plotting)
    plot_task_encoder_avgs = task_encoder_avgs
    plot_task_winners = task_winners
    if args.filter_winner:
        # Validate the encoder exists
        if args.filter_winner not in encoders_being_compared:
            print(f"Warning: Filter encoder '{args.filter_winner}' not found in results.")
            print(f"Available encoders: {', '.join(sorted(encoders_being_compared))}")
        else:
            # Filter to tasks where this encoder wins
            filtered_tasks = {
                task for task, winner in task_winners.items()
                if winner == args.filter_winner
            }
            plot_task_encoder_avgs = {
                task: avgs for task, avgs in task_encoder_avgs.items()
                if task in filtered_tasks
            }
            plot_task_winners = {
                task: winner for task, winner in task_winners.items()
                if task in filtered_tasks
            }
            print(f"\nFiltering plots to {len(filtered_tasks)} task(s) where '{args.filter_winner}' wins")
            if not filtered_tasks:
                print(f"Warning: '{args.filter_winner}' does not win any tasks. No plots will be generated.")

    # Print results
    print_results(
        task_encoder_avgs,
        wins_count,
        task_winners,
        args.metric,
        args.mode,
        args.places,
        args.methods,
        non_applicable
    )

    # Generate bar plot if requested
    if args.plot and plot_task_encoder_avgs:
        plot_encoder_comparison(
            plot_task_encoder_avgs,
            plot_task_winners,
            args.metric,
            args.mode,
            args.plot,
            args.methods,
            figsize=(args.plot_width, args.plot_height),
            dpi=args.plot_dpi
        )

    # Generate polar plot if requested
    if args.polar_plot and plot_task_encoder_avgs:
        plot_polar_encoder_comparison(
            plot_task_encoder_avgs,
            plot_task_winners,
            args.metric,
            args.mode,
            args.polar_plot,
            args.methods,
            figsize=(args.polar_size, args.polar_size),
            dpi=args.plot_dpi,
            fill_alpha=args.polar_fill_alpha,
            font_scale=args.polar_font_scale,
            zoom_axis=not args.polar_no_zoom,
            zoom_padding=args.polar_zoom_padding
        )

    # Save to JSON if requested
    if args.save_json:
        output = {
            'metric': args.metric,
            'mode': args.mode,
            'encoders_compared': sorted(results.keys()) if args.encoders else 'all',
            'methods_averaged': sorted(args.methods) if args.methods else 'all',
            'excluded_tasks': sorted([f"{dataset}/{task}" for dataset, task in excluded_tasks]) if excluded_tasks else [],
            'task_encoder_avgs': {
                f"{dataset}/{task}": {
                    encoder: {'mean': avg, 'num_methods': num_methods}
                    for encoder, (avg, num_methods) in encoder_avgs.items()
                }
                for (dataset, task), encoder_avgs in task_encoder_avgs.items()
            },
            'wins_count': wins_count,
            'task_winners': {
                f"{dataset}/{task}": winner
                for (dataset, task), winner in task_winners.items()
            },
            'non_applicable': {
                f"{dataset}/{task}": encoders
                for (dataset, task), encoders in non_applicable.items()
            } if non_applicable else {}
        }

        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved detailed results to: {args.save_json}")

    # Generate LaTeX table (enabled by default, disable with --no-latex)
    if args.latex and not args.no_latex:
        metric_data = collect_metric_data_with_std(
            results, args.table_metrics, args.methods
        )

        # Apply same task filtering as used for plots / prints
        if included_tasks is not None:
            metric_data = {
                k: v for k, v in metric_data.items()
                if k in included_tasks
            }
        if excluded_tasks:
            metric_data = {
                k: v for k, v in metric_data.items()
                if k not in excluded_tasks
            }

        if metric_data:
            encoder_list = args.encoders if args.encoders else sorted(results.keys())
            latex_content = generate_latex_table(
                metric_data,
                encoder_list,
                args.table_metrics,
                mode=args.mode,
                places=args.places,
                methods_used=args.methods,
            )

            tex_path = Path(args.latex)
            if not tex_path.suffix:
                tex_path = tex_path.with_suffix('.tex')
            tex_path.parent.mkdir(parents=True, exist_ok=True)

            with open(tex_path, 'w') as f:
                f.write(latex_content)
            print(f"\nLaTeX table saved to: {tex_path}")

            pdf_path = compile_latex_to_pdf(str(tex_path))
            if pdf_path:
                print(f"PDF compiled: {pdf_path}")
        else:
            print("No data available for LaTeX table after filtering.")


if __name__ == '__main__':
    main()
