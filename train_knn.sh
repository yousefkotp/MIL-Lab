#!/bin/bash
# k-NN Slurm job (mirrors train_linear.sh structure; CPU-bound, GPU not required)
#
# Environment overrides (exported by submitter or set inline before sbatch):
#   - REPO_ROOT: repo root path (defaults to SLURM_SUBMIT_DIR or PWD)
#   - FEATURES_SRC_DIR: absolute path to per-WSI vector features (.h5/.hdf5 with dataset 'features')
#   - FEATURES_PARENT_DIR: name of the parent directory containing .h5 files (e.g., features_lunit-vits8)
#   - DATASET: dataset name (derived from CSV_PATH if unset)
#   - TASK: task name (derived from CSV_PATH if unset)
#   - OUTPUT_DIR: override final output directory (default: results/<features_base>/<dataset>/<task>/knn)
#   - CSV_PATH: single CSV containing columns filename,label[,case_id]
#   - NUM_FOLDS: number of folds to create from CSV_PATH (default: 5)
#   - NORMALIZE: if set to 1/true, apply per-sample L2 normalization before k-NN
#   - CASE_FUSION: late or early fusion when multiple slides belong to the same patient (default: late)
#   - MAX_K: upper bound for auto-generated k grid (default: 101)
#   - KNN_WEIGHTS: scikit-learn weights argument (default: distance)
#   - KNN_METRIC: scikit-learn metric argument (default: minkowski)
#   - KNN_P: Minkowski power parameter (default: 2)
#   - KNN_ALGO: Neighbor search algorithm (default: brute)

#SBATCH -J train_knn
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 6
#SBATCH --time=05:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/output/%x_%j.txt
#SBATCH --error=logs/error/%x_%j.txt
#SBATCH --account=def-msh-ab
#SBATCH --exclude=fc11016

set -euo pipefail

# Resolve and move to repo root (prefer exported REPO_ROOT)
REPO_ROOT_DIR="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
cd "${REPO_ROOT_DIR}"
export PYTHONPATH="${REPO_ROOT_DIR}:${PYTHONPATH:-}"

# Ensure deterministic CuBLAS choice when PyTorch enables deterministic algorithms
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

FEATURES_SRC_DIR="${FEATURES_SRC_DIR:-}"
FEATURES_PARENT_DIR="${FEATURES_PARENT_DIR:-}"
CSV_PATH="${CSV_PATH:-}"
NUM_FOLDS="${NUM_FOLDS:-5}"

if [[ -z "${FEATURES_SRC_DIR}" ]]; then
  echo "ERROR: FEATURES_SRC_DIR is not set. Provide absolute path to vector features." >&2
  exit 1
fi
if [[ ! -d "${FEATURES_SRC_DIR}" ]]; then
  echo "ERROR: FEATURES_SRC_DIR does not exist or is not a directory: ${FEATURES_SRC_DIR}" >&2
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
if [[ -z "${FEATURES_PARENT_DIR}" ]]; then
  echo "ERROR: FEATURES_PARENT_DIR is not set. Provide the directory name containing .h5 features (e.g., features_lunit-vits8)." >&2
  exit 1
fi

CSV_DIR="$(dirname "${CSV_PATH}")"
DEFAULT_TASK="$(basename "${CSV_DIR}")"
DEFAULT_DATASET="$(basename "$(dirname "${CSV_DIR}")")"
DATASET="${DATASET:-${DEFAULT_DATASET}}"
TASK="${TASK:-${DEFAULT_TASK}}"
CONFIG_PATH="${CONFIG_PATH:-${CSV_DIR}/config.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: Required config.yaml not found at ${CONFIG_PATH}" >&2
  exit 1
fi

FEATURES_SRC_DIR="${FEATURES_SRC_DIR%/}"
FEATURES_BASENAME="$(basename "${FEATURES_SRC_DIR}")"

OUTPUT_DIR="${OUTPUT_DIR:-results/${FEATURES_BASENAME}/${DATASET}/${TASK}/knn}"
if [[ "${OUTPUT_DIR}" = /* ]]; then
  FINAL_OUTPUT_DIR="${OUTPUT_DIR}"
else
  FINAL_OUTPUT_DIR="${PWD%/}/${OUTPUT_DIR}"
fi

NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-6}}"
NORMALIZE="${NORMALIZE:-1}"
CASE_FUSION="${CASE_FUSION:-late}"
MAX_K="${MAX_K:-101}"
KNN_WEIGHTS="${KNN_WEIGHTS:-distance}"
KNN_METRIC="${KNN_METRIC:-minkowski}"
KNN_P="${KNN_P:-2}"
KNN_ALGO="${KNN_ALGO:-brute}"

################ ENFORCE SLURM TMPDIR ################
if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Error: SLURM_TMPDIR is not set. This script requires a Slurm temporary directory." >&2
  exit 2
fi

RUN_TMPDIR="${SLURM_TMPDIR%/}/knn_${SLURM_JOB_ID:-$$}"
TMP_FEATURES_DIR="${RUN_TMPDIR}/features"
TMP_OUTPUT_DIR="${RUN_TMPDIR}/output"

echo "Using SLURM_TMPDIR: ${SLURM_TMPDIR}"
echo "Run scratch: ${RUN_TMPDIR}"
echo "CSV path: ${CSV_PATH}"
echo "Features parent dir: ${FEATURES_PARENT_DIR}"
echo "Config path: ${CONFIG_PATH}"
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

echo "Collecting required feature files (parent dir=${FEATURES_PARENT_DIR})..."
FEATURE_MANIFEST="${RUN_TMPDIR}/feature_manifest.tsv"
FEATURE_MANIFEST_ERR="${RUN_TMPDIR}/feature_manifest.err"
set +e
python scripts/build_feature_manifest.py \
  --csv_path "${CSV_PATH}" \
  --config_path "${CONFIG_PATH}" \
  --features_dir "${FEATURES_SRC_DIR}" \
  --parent_dir "${FEATURES_PARENT_DIR}" \
  --mode knn \
  > "${FEATURE_MANIFEST}" 2> "${FEATURE_MANIFEST_ERR}"
MANIFEST_STATUS=$?
set -e
if [[ ${MANIFEST_STATUS} -ne 0 ]]; then
  echo "Error: Failed to build feature manifest. See ${FEATURE_MANIFEST_ERR} for details." >&2
  cat "${FEATURE_MANIFEST_ERR}" >&2 || true
  exit 1
fi

if [[ ! -s "${FEATURE_MANIFEST}" ]]; then
  echo "Error: Feature manifest is empty; no files selected for copy." >&2
  [[ -s "${FEATURE_MANIFEST_ERR}" ]] && cat "${FEATURE_MANIFEST_ERR}" >&2 || true
  exit 1
fi

mapfile -t FEATURE_LINES < "${FEATURE_MANIFEST}"
if [[ ${#FEATURE_LINES[@]} -eq 0 ]]; then
  echo "Error: No feature files found for CSV ${CSV_PATH} under ${FEATURES_SRC_DIR} with parent ${FEATURES_PARENT_DIR}" >&2
  [[ -s "${FEATURE_MANIFEST_ERR}" ]] && cat "${FEATURE_MANIFEST_ERR}" >&2 || true
  exit 1
fi
echo "Manifest built with ${#FEATURE_LINES[@]} files."
cp "${FEATURE_MANIFEST}" "${TMP_OUTPUT_DIR}/feature_manifest.tsv" || true

COPIED=0
for line in "${FEATURE_LINES[@]}"; do
  IFS=$'\t' read -r src rel <<<"${line}"
  dest="${TMP_FEATURES_DIR%/}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  cp "${src}" "${dest}"
  ((++COPIED))
done
echo "Feature subset copy complete: ${COPIED} files staged."

EXTRA_FLAGS=()
if [[ "${NORMALIZE}" == "1" || "${NORMALIZE,,}" == "true" ]]; then
  EXTRA_FLAGS+=("--normalize")
fi
if [[ -n "${CASE_FUSION}" ]]; then
  EXTRA_FLAGS+=("--case_fusion" "${CASE_FUSION}")
fi

echo "Starting k-NN training (outputs under ${TMP_OUTPUT_DIR})"
python train_knn.py \
  --csv_path "${CSV_PATH}" \
  --config_path "${CONFIG_PATH}" \
  --num_folds "${NUM_FOLDS}" \
  --features_dir "${TMP_FEATURES_DIR}" \
  --feature_parent_dir "${FEATURES_PARENT_DIR}" \
  --output_dir "${TMP_OUTPUT_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  --weights "${KNN_WEIGHTS}" \
  --metric "${KNN_METRIC}" \
  --p "${KNN_P}" \
  --algorithm "${KNN_ALGO}" \
  --max_k "${MAX_K}" \
  "${EXTRA_FLAGS[@]}"

# Stage-back handled by EXIT trap
