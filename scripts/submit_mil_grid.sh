#!/bin/bash
set -euo pipefail

DATASET="${DATASET:-CAMELYON17}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

# Models to run
MODELS=(
  abmil.base.slide_hubert_base.none
  clam.base_subtyping.slide_hubert_base.none
  dftd.base.slide_hubert_base.none
  dsmil.base.slide_hubert_base.none
  ilra.base.slide_hubert_base.none
  meanmil.base.slide_hubert_base.none
  rrt.base.slide_hubert_base.none
  transformer.base.slide_hubert_base.none
  transmil.base.slide_hubert_base.none
  wikg.base.slide_hubert_base.none
)

# MODELS=(
#   abmil.base.slide_hubert_base.none
#   clam.base_subtyping.slide_hubert_base.none
#   dftd.base.slide_hubert_base.none
#   dsmil.base.slide_hubert_base.none
#   ilra.base.slide_hubert_base.none
#   meanmil.base.slide_hubert_base.none
#   rrt.base.slide_hubert_base.none
#   transformer.base.slide_hubert_base.none
#   transmil.base.slide_hubert_base.none
#   wikg.base.slide_hubert_base.none
# )

# Feature directories to iterate (absolute paths)
FEATURE_DIRS=(
  /home/mila/k/kotpy/scratch/datasets/CAMELYON17/slide_hubert/mean
  /home/mila/k/kotpy/scratch/datasets/CAMELYON17/slide_hubert/mean_l2
  /home/mila/k/kotpy/scratch/datasets/CAMELYON17/slide_hubert/cls
  /home/mila/k/kotpy/scratch/datasets/CAMELYON17/slide_hubert/cls_l2
)

# uni_v1
# uni_v2
# conch_v1
# conch_v15
# gigapath
# hibou_b
# hibou_l
# resnet50
# hoptimus_0
# hoptimus_1
# midnight12k
# phikon_v1
# phikon_v2
# virchow
# virchow2
# dino_vit_small_p8_embeddings

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
LAUNCH_SCRIPT="${REPO_ROOT}/train_mil.sh"

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  echo "train_mil.sh not found at ${LAUNCH_SCRIPT}" >&2
  exit 1
fi

echo "Submitting jobs for ${#MODELS[@]} models × ${#FEATURE_DIRS[@]} feature sets..."

for fdir in "${FEATURE_DIRS[@]}"; do
  fdir_noslash="${fdir%/}"
  feat_base="$(basename "${fdir_noslash}")"
  # Optional existence check (non-fatal)
  if [[ ! -d "${fdir_noslash}" ]]; then
    echo "Warning: features dir not found: ${fdir_noslash}" >&2
  fi

  for model in "${MODELS[@]}"; do
    model_prefix="${model%%.*}"
    model_second="$(echo "${model}" | cut -d. -f2)"
    if [[ "${model_prefix}" == "dftd" && "${model_second}" == "base_afs" ]]; then
      model_dir_name="dftd_afs"
    else
      model_dir_name="${model_prefix}"
    fi

    job_name="train_${model_dir_name}_slide_hubert_${feat_base}"
    out_dir="results/${feat_base}/${DATASET}/${model_dir_name}"

    echo "sbatch --job-name ${job_name} (features=${feat_base}, model=${model})"
    sbatch --job-name "${job_name}" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",FEATURES_SRC_DIR="${fdir_noslash}",MODEL="${model}",DATASET="${DATASET}",OUTPUT_DIR="${out_dir}",GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
      "${LAUNCH_SCRIPT}"
  done
done

echo "All submissions attempted."
