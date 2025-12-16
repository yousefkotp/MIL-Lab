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
import numpy as np


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
    all_encoders = sorted(set(
        encoder
        for encoder_avgs in task_encoder_avgs.values()
        for encoder in encoder_avgs.keys()
    ))

    # Calculate wins for each encoder
    wins_count = defaultdict(int)
    for winner in task_winners.values():
        wins_count[winner] += 1
    total_tasks = len(task_winners)

    # Prepare data for plotting
    n_tasks = len(sorted_tasks)
    n_encoders = len(all_encoders)

    # Create color map for encoders
    colors = plt.cm.Set2(np.linspace(0, 1, n_encoders))
    encoder_colors = {encoder: colors[i] for i, encoder in enumerate(all_encoders)}

    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set bar width and positions
    bar_width = 0.8 / n_encoders
    x = np.arange(n_tasks)

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
                     color=encoder_colors[encoder], alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add star markers for winners
        for j, task in enumerate(sorted_tasks):
            if task_winners.get(task) == encoder and values[j] > 0:
                ax.plot(x[j] + offset, values[j], marker='*', markersize=15,
                       color='gold', markeredgecolor='black', markeredgewidth=0.5, zorder=10)

    # Customize plot
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric}', fontsize=12, fontweight='bold')

    mode_text = 'higher is better' if mode == 'max' else 'lower is better'
    title = f'Encoder Comparison Across Tasks\n{metric} ({mode_text})'
    # if methods_used:
    #     title += f"\nMethods averaged: {', '.join(sorted(methods_used))}"
    # else:
    #     title += "\nMethods averaged: all available"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set x-axis labels with color coding by dataset
    task_labels = [f"{dataset}/{task}" for dataset, task in sorted_tasks]
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=90, ha='right', fontsize=9)

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
            ax.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
        prev_dataset = dataset

    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10,
             title='Encoders', title_fontsize=11, framealpha=0.9)

    # Add note about stars and color coding
    notes = '★ = Winner for this task\nX-axis labels colored by dataset'
    fig.text(0.99, 0.01, notes,
            ha='right', va='bottom', fontsize=9, style='italic')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to: {output_path}")


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
        default=4,
        help='Decimal places for output (default: 4)'
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
        default="compare.png",
        help='Save bar plot to specified file path (e.g., encoder_comparison.png)'
    )
    parser.add_argument(
        '--plot-width',
        type=int,
        default=20,
        help='Plot width in inches (default: 20)'
    )
    parser.add_argument(
        '--plot-height',
        type=int,
        default=8,
        help='Plot height in inches (default: 8)'
    )
    parser.add_argument(
        '--plot-dpi',
        type=int,
        default=300,
        help='Plot DPI (default: 300)'
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

    # Generate plot if requested
    if args.plot:
        plot_encoder_comparison(
            task_encoder_avgs,
            task_winners,
            args.metric,
            args.mode,
            args.plot,
            args.methods,
            figsize=(args.plot_width, args.plot_height),
            dpi=args.plot_dpi
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


if __name__ == '__main__':
    main()
