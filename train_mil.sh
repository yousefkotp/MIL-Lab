#!/bin/bash
#SBATCH -J train_abmil_slide_hubert_5000_10000
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/mila/k/kotpy/scratch/MIL-Lab/logs/output/%x_%j.txt
#SBATCH --error=/home/mila/k/kotpy/scratch/MIL-Lab/logs/error/%x_%j.txt
#SBATCH --mem=64G

set -euo pipefail

# Ensure we run from the repository root
# - When sbatch copies the script to the slurm spool, "$0" points there,
#   so we can't reliably use dirname "$0". Prefer an exported REPO_ROOT
#   from the submit script, else fall back to SLURM_SUBMIT_DIR.
REPO_ROOT_DIR="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
cd "${REPO_ROOT_DIR}"
#
# Config (overridable via environment)
# - MODEL: full model identifier (e.g., abmil.base.slide_hubert.none)
# - FEATURES_SRC_DIR: absolute path to features directory to copy from
# - DATASET: dataset name used to build default output path (default: CAMELYON17)
# - OUTPUT_DIR: if set, overrides the auto-constructed results path
#
MODEL="${MODEL:-abmil.base.slide_hubert.none}"
FEATURES_SRC_DIR="${FEATURES_SRC_DIR:-/home/mila/k/kotpy/scratch/datasets/CAMELYON17/slide_hubert/20x_512px/clusters_5000_10000_steps_20K_average_overlaps}"
DATASET="${DATASET:-CAMELYON17}"

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
OUTPUT_DIR="${OUTPUT_DIR:-results/slide_hubert/${FEATURES_BASENAME}/${DATASET}/${MODEL_DIR_NAME}}"

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

echo "Copying features from ${FEATURES_SRC_DIR} to ${TMP_FEATURES_DIR} ..."
# Copy contents (not the directory wrapper)
cp -a "${FEATURES_SRC_DIR%/}/." "${TMP_FEATURES_DIR}/"
echo "Feature copy complete."

# CPU workers: default to SLURM_CPUS_PER_TASK if set, else 6
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-6}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

echo "Starting training (writing outputs under ${TMP_OUTPUT_DIR})"
python train_mil.py \
    --csv_paths /home/mila/k/kotpy/scratch/softpatchify_code/csvs/CAMELYON17/fold_1.csv /home/mila/k/kotpy/scratch/softpatchify_code/csvs/CAMELYON17/fold_2.csv /home/mila/k/kotpy/scratch/softpatchify_code/csvs/CAMELYON17/fold_3.csv /home/mila/k/kotpy/scratch/softpatchify_code/csvs/CAMELYON17/fold_4.csv /home/mila/k/kotpy/scratch/softpatchify_code/csvs/CAMELYON17/fold_5.csv \
    --features_dir "${TMP_FEATURES_DIR}" \
    --model "${MODEL}" \
    --num_workers "${NUM_WORKERS}" \
    --output_dir "${TMP_OUTPUT_DIR}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --balanced_sampling

# Stage-back also happens through the EXIT trap; nothing else to do.
