#!/bin/bash
#SBATCH -J train_mil
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/output/%x_%j.txt
#SBATCH --error=logs/error/%x_%j.txt
#SBATCH --mem=64G
#SBATCH --account=def-msh-ab
#SBATCH --exclude=fc11016

set -euo pipefail

# Ensure we run from the repository root
# - When sbatch copies the script to the slurm spool, "$0" points there,
#   so we can't reliably use dirname "$0". Prefer an exported REPO_ROOT
#   from the submit script, else fall back to SLURM_SUBMIT_DIR.
REPO_ROOT_DIR="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
cd "${REPO_ROOT_DIR}"
export PYTHONPATH="${REPO_ROOT_DIR}:${PYTHONPATH:-}"
#
# Config (overridable via environment)
# - MODEL: full model identifier (e.g., abmil.base.slide_hubert.none)
# - FEATURES_SRC_DIR: absolute path to features directory to copy from
# - DATASET: dataset name used to build default output path (derived from CSV_PATH if unset)
# - TASK: task name used in output path (derived from CSV_PATH if unset)
# - OUTPUT_DIR: if set, overrides the auto-constructed results path
# - CSV_PATH: path to the single CSV containing columns case_id (optional), filename, label
# - NUM_FOLDS: number of folds to create from CSV (default: 5)
#
MODEL="${MODEL:-}"
FEATURES_SRC_DIR="${FEATURES_SRC_DIR:-}"
FEATURES_PARENT_DIR="${FEATURES_PARENT_DIR:-}"
CSV_PATH="${CSV_PATH:-}"
NUM_FOLDS="${NUM_FOLDS:-5}"

if [[ -z "${CSV_PATH}" ]]; then
  echo "Error: CSV_PATH is not set. Provide path to CSV with filename,label[,case_id]." >&2
  exit 1
fi
if [[ ! -f "${CSV_PATH}" ]]; then
  echo "Error: CSV_PATH does not exist: ${CSV_PATH}" >&2
  exit 1
fi
if [[ -z "${FEATURES_PARENT_DIR}" ]]; then
  echo "Error: FEATURES_PARENT_DIR is not set. Provide the directory name containing .h5 features (e.g., features_lunit-vits8)." >&2
  exit 1
fi
if [[ -z "${FEATURES_SRC_DIR}" ]]; then
  echo "Error: FEATURES_SRC_DIR is not set. Provide absolute path to the feature root directory." >&2
  exit 1
fi
if [[ ! -d "${FEATURES_SRC_DIR}" ]]; then
  echo "Error: FEATURES_SRC_DIR does not exist or is not a directory: ${FEATURES_SRC_DIR}" >&2
  exit 1
fi

# Derive dataset/task from CSV path if not provided
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

# Normalize and derive names
FEATURES_SRC_DIR="${FEATURES_SRC_DIR%/}"
FEATURES_BASENAME="$(basename "${FEATURES_SRC_DIR}")"
# Derive a directory-friendly model name, disambiguating dftd.base vs dftd.base_afs
MODEL_NAME_PREFIX="${MODEL%%.*}"
MODEL_SECOND_PART="$(echo "${MODEL}" | cut -d. -f2)"
if [[ "${MODEL_NAME_PREFIX}" == "dftd" && "${MODEL_SECOND_PART}" == "base_afs" ]]; then
  MODEL_DIR_NAME="dftd_afs"
else
  MODEL_DIR_NAME="${MODEL_NAME_PREFIX}"
fi

# Default results directory if not explicitly provided
OUTPUT_DIR="${OUTPUT_DIR:-results/${FEATURES_BASENAME}/${DATASET}/${TASK}/${MODEL_DIR_NAME}}"

# Normalize final destination output directory to absolute path rooted at repo if relative
if [[ "${OUTPUT_DIR}" = /* ]]; then
  FINAL_OUTPUT_DIR="${OUTPUT_DIR}"
else
  FINAL_OUTPUT_DIR="${PWD%/}/${OUTPUT_DIR}"
fi

################ ENFORCE SLURM TMPDIR ################
if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Error: SLURM_TMPDIR is not set. This script requires a Slurm temporary directory." >&2
  exit 2
fi

# Create a job-scoped run directory inside SLURM_TMPDIR for isolation
RUN_TMPDIR="${SLURM_TMPDIR%/}/mil_${SLURM_JOB_ID:-$$}"
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

# Always stage back outputs on exit, even on errors (best effort)
stage_back() {
  if [[ -d "${TMP_OUTPUT_DIR}" ]]; then
    echo "Staging outputs from ${TMP_OUTPUT_DIR} -> ${FINAL_OUTPUT_DIR}"
    mkdir -p "${FINAL_OUTPUT_DIR}"
    # Best-effort copy; don't fail the trap if copy has issues
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
  --mode mil \
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
# Keep a copy of the manifest for debugging (staged back with outputs)
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

# CPU workers: default to SLURM_CPUS_PER_TASK if set, else 6
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-6}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EXTRA_FLAGS=(--balanced_sampling)

echo "Starting training (writing outputs under ${TMP_OUTPUT_DIR})"
python train_mil.py \
    --csv_path "${CSV_PATH}" \
    --config_path "${CONFIG_PATH}" \
    --num_folds "${NUM_FOLDS}" \
    --features_dir "${TMP_FEATURES_DIR}" \
    --feature_parent_dir "${FEATURES_PARENT_DIR}" \
    --model "${MODEL}" \
    --num_workers "${NUM_WORKERS}" \
    --output_dir "${TMP_OUTPUT_DIR}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    "${EXTRA_FLAGS[@]}"

# Stage-back also happens through the EXIT trap; nothing else to do.
