#!/bin/bash
set -euo pipefail

# Define tasks (format: .../<dataset>/<task>/task.csv)
CSV_PATHS=(
  /home/kotpaz/scratch/tasks/custom/bc_therapy/grade/task.csv
)
if [[ ${#CSV_PATHS[@]} -eq 0 ]]; then
  echo "CSV_PATHS must list at least one task CSV (format: .../<dataset>/<task>/task.csv)." >&2
  exit 1
fi
for csvp in "${CSV_PATHS[@]}"; do
  if [[ ! -f "${csvp}" ]]; then
    echo "CSV path not found: ${csvp}" >&2
    exit 1
  fi
done

# Hyperparams (overridable)
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-6}"
BALANCED_SAMPLING="${BALANCED_SAMPLING:-1}"
NORMALIZE="${NORMALIZE:-0}"
NUM_FOLDS="${NUM_FOLDS:-5}"

# Feature directories to iterate (absolute paths)
# Update these with the actual per-WSI vector features ('.h5'/'hdf5' with dataset 'features')
FEATURE_DIRS=(
  /home/mila/k/kotpy/scratch/datasets/CAMELYON17/trident/20x_512px_0px_overlap/slide_features_titan
  /home/mila/k/kotpy/scratch/datasets/CAMELYON17/trident/20x_512px_0px_overlap/slide_features_feather
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
LAUNCH_SCRIPT="${REPO_ROOT}/train_linear.sh"

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  echo "train_linear.sh not found at ${LAUNCH_SCRIPT}" >&2
  exit 1
fi

echo "Submitting linear jobs for ${#FEATURE_DIRS[@]} feature sets over ${#CSV_PATHS[@]} tasks..."

for csv_path in "${CSV_PATHS[@]}"; do
  csv_dir="$(dirname "${csv_path}")"
  task_name="$(basename "${csv_dir}")"
  dataset_name="$(basename "$(dirname "${csv_dir}")")"

  for fdir in "${FEATURE_DIRS[@]}"; do
    fdir_noslash="${fdir%/}"
    feat_base="$(basename "${fdir_noslash}")"
    if [[ ! -d "${fdir_noslash}" ]]; then
      echo "Warning: features dir not found: ${fdir_noslash}" >&2
    fi

    job_name="train_linear_${feat_base}_${dataset_name}_${task_name}"
    out_dir="results/${feat_base}/${dataset_name}/${task_name}/linear"

    echo "sbatch --job-name ${job_name} (features=${feat_base}, dataset=${dataset_name}, task=${task_name})"
    sbatch --job-name "${job_name}" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",FEATURES_SRC_DIR="${fdir_noslash}",DATASET="${dataset_name}",TASK="${task_name}",OUTPUT_DIR="${out_dir}",EPOCHS="${EPOCHS}",LR="${LR}",WEIGHT_DECAY="${WEIGHT_DECAY}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}",BALANCED_SAMPLING="${BALANCED_SAMPLING}",NORMALIZE="${NORMALIZE}",CSV_PATH="${csv_path}",NUM_FOLDS="${NUM_FOLDS}" \
      "${LAUNCH_SCRIPT}"
  done
done

echo "All submissions attempted."
