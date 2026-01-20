#!/bin/bash
set -euo pipefail

# Define tasks (format: .../<dataset>/<task>/task.csv)
CSV_PATHS=(
  /home/kotpaz/scratch/tasks/custom/TCGA-BRCA/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-COAD/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-ESCA/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-SARC/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-TGCT/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-THYM/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-UCEC/primary_diagnosis/task.csv
)

# CSV_PATHS=(
#   /home/kotpaz/scratch/tasks/custom/bc_therapy/er_status/task.csv
#   /home/kotpaz/scratch/tasks/custom/bc_therapy/grade/task.csv
#   /home/kotpaz/scratch/tasks/custom/bc_therapy/her2_status/task.csv
#   /home/kotpaz/scratch/tasks/custom/bc_therapy/residual_cancer_burden/task.csv
#   /home/kotpaz/scratch/tasks/custom/bracs/coarse/task.csv
#   /home/kotpaz/scratch/tasks/custom/bracs/fine/task.csv
#   /home/kotpaz/scratch/tasks/custom/camelyon17/breast_cancer_metastases/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_brca/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_brca/PIK3CA_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_brca/TP53_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_ccrcc/BAP1_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_ccrcc/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_ccrcc/PBRM1_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_ccrcc/VHL_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_coad/ACVR2A_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_coad/APC_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_coad/ARID1A_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_coad/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_coad/KRAS_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_coad/MSI_H/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_coad/PIK3CA_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_coad/SETD1B_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_coad/TP53_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_gbm/EGFR_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_gbm/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_gbm/TP53_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_hnsc/CASP8_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_hnsc/Histologic_Grade/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_hnsc/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_lscc/ARID1A_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_lscc/Histologic_Grade/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_lscc/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_lscc/KEAP1_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_luad/EGFR_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_luad/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_luad/KRAS_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_luad/STK11_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_luad/TP53_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_ov/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_pda/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_pda/SMAD4_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_ucec/CTNNB1_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_ucec/Immune_class/task.csv
#   /home/kotpaz/scratch/tasks/custom/cptac_ucec/PTEN_mutation/task.csv
#   /home/kotpaz/scratch/tasks/custom/dhmc_kidney/morphological_subtyping/task.csv
#   /home/kotpaz/scratch/tasks/custom/dhmc_luad/histologic_pattern/task.csv
#   /home/kotpaz/scratch/tasks/custom/ebrains/diagnosis/task.csv
#   /home/kotpaz/scratch/tasks/custom/ebrains/diagnosis_group/task.csv
#   /home/kotpaz/scratch/tasks/custom/ebrains/idh_status/task.csv
#   /home/kotpaz/scratch/tasks/custom/imp/grade/task.csv
#   /home/kotpaz/scratch/tasks/custom/mbc/treatment_response/task.csv
#   /home/kotpaz/scratch/tasks/custom/panda/prostate_cancer_grade/task.csv
#   /home/kotpaz/scratch/tasks/custom/imp_cervix/dysplasia_grading/task.csv
# )
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

GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_FOLDS="${NUM_FOLDS:-5}"

# Models to run
MODELS=(
  abmil.base.slide_hubert_base.none
  clam.base_subtyping.slide_hubert_base.none
  dsmil.base.slide_hubert_base.none
  meanmil.base.slide_hubert_base.none
  transmil.base.slide_hubert_base.none
)
# MODELS=(
#   abmil.base.slide_hubert_base.none
#   clam.base_subtyping.slide_hubert_base.none
#   dsmil.base.slide_hubert_base.none
#   meanmil.base.slide_hubert_base.none
#   transmil.base.slide_hubert_base.none
# )

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
  /home/kotpaz/projects/rrg-msh/kotpaz/datasets/backbone_embeddings
)
FEATURES_PARENT_DIR="features_lunit-vits8"
FEAT_BASE_NAME="backbone_embeddings"  # Hardcoded name for output directory

# /home/kotpaz/projects/rrg-msh/kotpaz/datasets/iBOT_training/trained_models/0x_cls_mean_11000_16x16_step_16x16

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

echo "Submitting jobs for ${#MODELS[@]} models × ${#FEATURE_DIRS[@]} feature sets × ${#CSV_PATHS[@]} tasks..."

for fdir in "${FEATURE_DIRS[@]}"; do
  fdir_noslash="${fdir%/}"
  feat_base="${FEAT_BASE_NAME}"
  # Optional existence check (non-fatal)
  if [[ ! -d "${fdir_noslash}" ]]; then
    echo "Warning: features dir not found: ${fdir_noslash}" >&2
  fi

  for csv_path in "${CSV_PATHS[@]}"; do
    csv_dir="$(dirname "${csv_path}")"
    task_name="$(basename "${csv_dir}")"
    dataset_name="$(basename "$(dirname "${csv_dir}")")"

    for model in "${MODELS[@]}"; do
      model_prefix="${model%%.*}"
      model_second="$(echo "${model}" | cut -d. -f2)"
      if [[ "${model_prefix}" == "dftd" && "${model_second}" == "base_afs" ]]; then
        model_dir_name="dftd_afs"
      else
        model_dir_name="${model_prefix}"
      fi

      job_name="train_${model_dir_name}_${feat_base}_${dataset_name}_${task_name}"
      out_dir="results/${feat_base}/${dataset_name}/${task_name}/${model_dir_name}"

      echo "sbatch --job-name ${job_name} (features=${feat_base}, dataset=${dataset_name}, task=${task_name}, model=${model})"
      sbatch --job-name "${job_name}" \
        --export=ALL,REPO_ROOT="${REPO_ROOT}",FEATURES_SRC_DIR="${fdir_noslash}",FEATURES_PARENT_DIR="${FEATURES_PARENT_DIR}",MODEL="${model}",DATASET="${dataset_name}",TASK="${task_name}",OUTPUT_DIR="${out_dir}",GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}",CSV_PATH="${csv_path}",NUM_FOLDS="${NUM_FOLDS}" \
        "${LAUNCH_SCRIPT}"
    done
  done
done

echo "All submissions attempted."
