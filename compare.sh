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

# Index by dataset/task/model -> list of feature dirs
declare -A GROUPS
for run in "${ALL_RUNS[@]}"; do
  # Extract components
  run_rel="${run#${RESULTS_ROOT%/}/}"
  IFS='/' read -r feature dataset task model <<< "${run_rel}"
  key="${dataset}/${task}/${model}"
  GROUPS["${key}"]+="${run} "
done

LOG_ROOT="${RESULTS_ROOT%/}/../logs/compare"
mkdir -p "${LOG_ROOT}"

for key in "${!GROUPS[@]}"; do
  dataset="${key%%/*}"
  rest="${key#*/}"
  task="${rest%%/*}"
  model="${rest#*/}"

  # Prepare outputs
  out_dir="${LOG_ROOT}/${dataset}/${task}"
  plots_dir="${out_dir}/${model}"
  latex_file="${out_dir}/${model}.tex"
  mkdir -p "${plots_dir}"

  # Collect paths and names
  paths=(${GROUPS["${key}"]})
  names=()
  for p in "${paths[@]}"; do
    feature="$(basename "$(dirname "$(dirname "$(dirname "${p}")")")")"
    names+=("${feature}")
  done

  echo "== Dataset: ${dataset} | Task: ${task} | Model: ${model} (features: ${#paths[@]}) =="
  python "${COMPARE_SCRIPT}" \
    "${paths[@]}" \
    --names "${names[@]}" \
    --format latex \
    --latex-file "${latex_file}" \
    --only-intersection \
    --latex-include-std \
    --latex-include-embeddings \
    --latex-keep-aux \
    --save-plots \
    --plots-dir "${plots_dir}"
done

echo "Comparison tables written under ${LOG_ROOT}"
