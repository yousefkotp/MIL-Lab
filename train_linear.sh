#!/bin/bash
# Linear probe Slurm job (mirrors train_mil.sh structure)
#
# Environment overrides (exported by submitter or set inline before sbatch):
#   - REPO_ROOT: repo root path (defaults to SLURM_SUBMIT_DIR or PWD)
#   - FEATURES_SRC_DIR: absolute path to per-WSI vector features (.h5/.hdf5 with dataset 'features')
#   - DATASET: dataset name (derived from CSV_PATH if unset)
#   - TASK: task name (derived from CSV_PATH if unset)
#   - OUTPUT_DIR: override final output directory (default: results/<features_base>/<dataset>/<task>/linear)
#   - EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, NUM_WORKERS: training hyperparameters
#   - BALANCED_SAMPLING: if set to 1/true, pass --balanced_sampling
#   - NORMALIZE: if set to 1/true, pass --normalize
#   - CSV_PATH: single CSV containing columns filename,label[,case_id]
#   - NUM_FOLDS: number of folds to create from CSV_PATH (default: 5)

#SBATCH -J train_linear_camelyon
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/mila/k/kotpy/scratch/MIL-Lab/logs/output/%x_%j.txt
#SBATCH --error=/home/mila/k/kotpy/scratch/MIL-Lab/logs/error/%x_%j.txt

set -euo pipefail

# Resolve and move to repo root (prefer exported REPO_ROOT)
REPO_ROOT_DIR="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
cd "${REPO_ROOT_DIR}"

# Ensure deterministic CuBLAS choice when PyTorch enables deterministic algorithms
# See: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

FEATURES_SRC_DIR="${FEATURES_SRC_DIR:-}"   # should be absolute path
CSV_PATH="${CSV_PATH:-}"  # required: single CSV with filename,label[,case_id]
NUM_FOLDS="${NUM_FOLDS:-5}"

if [[ -z "${FEATURES_SRC_DIR}" ]]; then
  echo "ERROR: FEATURES_SRC_DIR is not set. Provide absolute path to vector features." >&2
  exit 1
fi
if [[ -z "${CSV_PATH}" ]]; then
  echo "ERROR: CSV_PATH is not set. Provide path to CSV with filename,label[,case_id]." >&2
  exit 1
fi
if [[ ! -f "${CSV_PATH}" ]]; then
  echo "ERROR: CSV_PATH does not exist: ${CSV_PATH}" >&2
  exit 1
fi

# Derive dataset/task from CSV path if not provided
CSV_DIR="$(dirname "${CSV_PATH}")"
DEFAULT_TASK="$(basename "${CSV_DIR}")"
DEFAULT_DATASET="$(basename "$(dirname "${CSV_DIR}")")"
DATASET="${DATASET:-${DEFAULT_DATASET}}"
TASK="${TASK:-${DEFAULT_TASK}}"

# Normalize inputs
FEATURES_SRC_DIR="${FEATURES_SRC_DIR%/}"
FEATURES_BASENAME="$(basename "${FEATURES_SRC_DIR}")"

# Default outputs under repo, unless OUTPUT_DIR is provided (absolute allowed)
OUTPUT_DIR="${OUTPUT_DIR:-results/${FEATURES_BASENAME}/${DATASET}/${TASK}/linear}"
if [[ "${OUTPUT_DIR}" = /* ]]; then
  FINAL_OUTPUT_DIR="${OUTPUT_DIR}"
else
  FINAL_OUTPUT_DIR="${PWD%/}/${OUTPUT_DIR}"
fi

# Training hyperparams
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-6}}"

# Flags
BALANCED_SAMPLING="${BALANCED_SAMPLING:-0}"
NORMALIZE="${NORMALIZE:-1}"

################ ENFORCE SLURM TMPDIR ################
if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Error: SLURM_TMPDIR is not set. This script requires a Slurm temporary directory." >&2
  exit 2
fi

RUN_TMPDIR="${SLURM_TMPDIR%/}/linear_${SLURM_JOB_ID:-$$}"
TMP_FEATURES_DIR="${RUN_TMPDIR}/features"
TMP_OUTPUT_DIR="${RUN_TMPDIR}/output"

echo "Using SLURM_TMPDIR: ${SLURM_TMPDIR}"
echo "Run scratch: ${RUN_TMPDIR}"
echo "CSV path: ${CSV_PATH}"
echo "Dataset: ${DATASET} | Task: ${TASK}"
echo "Folds: ${NUM_FOLDS}"
mkdir -p "${TMP_FEATURES_DIR}" "${TMP_OUTPUT_DIR}"

stage_back() {
  if [[ -d "${TMP_OUTPUT_DIR}" ]]; then
    echo "Staging outputs from ${TMP_OUTPUT_DIR} -> ${FINAL_OUTPUT_DIR}"
    mkdir -p "${FINAL_OUTPUT_DIR}"
    cp -a "${TMP_OUTPUT_DIR%/}/." "${FINAL_OUTPUT_DIR%/}/" || true
    echo "Stage-back complete."
  fi
}
trap stage_back EXIT

echo "Copying features from ${FEATURES_SRC_DIR} to ${TMP_FEATURES_DIR} ..."
cp -a "${FEATURES_SRC_DIR%/}/." "${TMP_FEATURES_DIR}/"
echo "Feature copy complete."

# Build flags
EXTRA_FLAGS=()
if [[ "${BALANCED_SAMPLING}" == "1" || "${BALANCED_SAMPLING,,}" == "true" ]]; then
  EXTRA_FLAGS+=("--balanced_sampling")
fi
if [[ "${NORMALIZE}" == "1" || "${NORMALIZE,,}" == "true" ]]; then
  EXTRA_FLAGS+=("--normalize")
fi

echo "Starting linear training (outputs under ${TMP_OUTPUT_DIR})"
python train_linear.py \
  --csv_path "${CSV_PATH}" \
  --num_folds "${NUM_FOLDS}" \
  --features_dir "${TMP_FEATURES_DIR}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --output_dir "${TMP_OUTPUT_DIR}" \
  "${EXTRA_FLAGS[@]}"

# Note: Stage-back handled by EXIT trap
