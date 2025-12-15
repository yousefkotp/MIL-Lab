#!/bin/bash
# Auto-create comparison tables for all tasks under results/, grouped by dataset/task and grouped by MIL model.
#
# Usage: ./compare.sh [RESULTS_ROOT]
# Default RESULTS_ROOT: results
#
# For each task directory matching results/<feature>/<dataset>/<task>/<model>,
# we generate a LaTeX table comparing metrics across all feature sets for a
# single MIL model (i.e., fix <model> and vary <feature>).
#
# Outputs:
#   - Tables: logs/compare/<dataset>/<task>/<model>.tex
#   - Plots:  logs/compare/<dataset>/<task>/<model>/

set -euo pipefail

RESULTS_ROOT="${1:-results}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT="${SCRIPT_DIR}/scripts/compare_val_metrics.py"

if [[ ! -f "${COMPARE_SCRIPT}" ]]; then
  echo "compare_val_metrics.py not found at ${COMPARE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${RESULTS_ROOT}" ]]; then
  echo "Results root not found: ${RESULTS_ROOT}" >&2
  exit 1
fi

# Discover all paths of the form results/<feature>/<dataset>/<task>/<model>
mapfile -t ALL_RUNS < <(find "${RESULTS_ROOT}" -mindepth 4 -maxdepth 4 -type d)

if [[ ${#ALL_RUNS[@]} -eq 0 ]]; then
  echo "No runs found under ${RESULTS_ROOT}" >&2
  exit 0
fi

# Index runs
declare -A LINEAR_BY_TASK           # dataset/task -> linear path (first seen)
declare -A FEATURE_ROOTS            # dataset/task -> space-separated feature roots
for run in "${ALL_RUNS[@]}"; do
  run_rel="${run#${RESULTS_ROOT%/}/}"
  IFS='/' read -r feature dataset task model <<< "${run_rel}"
  # Track feature roots per task
  ft_key="${dataset}/${task}"
  # only record once per feature root
  case " ${FEATURE_ROOTS[${ft_key}]:-} " in
    *" ${feature} "* ) ;;
    * ) FEATURE_ROOTS["${ft_key}"]+="${feature} ";;
  esac
  # Track linear path per task (use first encountered)
  if [[ "${model}" == "linear" && -z "${LINEAR_BY_TASK[${ft_key}]:-}" ]]; then
    LINEAR_BY_TASK["${ft_key}"]="${run}"
  fi
done

LOG_ROOT="compare"
mkdir -p "${LOG_ROOT}"

# ---------------------------------------------------------------------------
# Also create per-task aggregate tables: all methods x all features together.
# ---------------------------------------------------------------------------
for ft_key in "${!FEATURE_ROOTS[@]}"; do
  dataset="${ft_key%%/*}"
  task="${ft_key#*/}"
  # Build roots array from feature roots, adding linear task root if present and not already included
  features=(${FEATURE_ROOTS["${ft_key}"]})
  roots=()
  for feat in "${features[@]}"; do
    roots+=("${RESULTS_ROOT%/}/${feat}/${dataset}/${task}")
  done
  if [[ -n "${LINEAR_BY_TASK["${ft_key}"]:-}" ]]; then
    lin_root="$(dirname "${LINEAR_BY_TASK["${ft_key}"]}")"
    if [[ " ${roots[*]} " != *" ${lin_root} "* ]]; then
      roots+=("${lin_root}")
      features+=("windows")
    fi
  fi
  out_dir="${LOG_ROOT}/${dataset}/${task}"
  mkdir -p "${out_dir}"
  latex_file="${out_dir}/all_methods.tex"
  plots_dir="${out_dir}/all_methods"
  mkdir -p "${plots_dir}"
  echo "== Aggregate table: ${dataset} | ${task} (features: ${#roots[@]}) =="
  python "${COMPARE_SCRIPT}" \
    "${roots[@]}" \
    --names "${features[@]}" \
    --format latex \
    --latex-file "${latex_file}" \
    --latex-compile \
    --latex-include-std \
    --latex-include-embeddings \
    --latex-keep-aux
done
