#!/bin/bash
set -euo pipefail

# Define tasks (format: .../<dataset>/<task>/task.csv)
CSV_PATHS=(
  /home/kotpaz/scratch/tasks/custom/bc_therapy/er_status/task.csv
  /home/kotpaz/scratch/tasks/custom/bc_therapy/grade/task.csv
  /home/kotpaz/scratch/tasks/custom/bc_therapy/her2_status/task.csv
  /home/kotpaz/scratch/tasks/custom/bc_therapy/residual_cancer_burden/task.csv
  /home/kotpaz/scratch/tasks/custom/bracs/coarse/task.csv
  /home/kotpaz/scratch/tasks/custom/bracs/fine/task.csv
  /home/kotpaz/scratch/tasks/custom/camelyon17/breast_cancer_metastases/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_brca/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_brca/PIK3CA_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_brca/TP53_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_ccrcc/BAP1_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_ccrcc/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_ccrcc/PBRM1_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_ccrcc/VHL_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_coad/ACVR2A_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_coad/APC_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_coad/ARID1A_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_coad/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_coad/KRAS_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_coad/MSI_H/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_coad/PIK3CA_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_coad/SETD1B_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_coad/TP53_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_gbm/EGFR_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_gbm/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_gbm/TP53_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_hnsc/CASP8_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_hnsc/Histologic_Grade/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_hnsc/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_lscc/ARID1A_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_lscc/Histologic_Grade/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_lscc/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_lscc/KEAP1_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_luad/EGFR_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_luad/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_luad/KRAS_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_luad/STK11_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_luad/TP53_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_ov/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_pda/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_pda/SMAD4_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_ucec/CTNNB1_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_ucec/Immune_class/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_ucec/PTEN_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/dhmc_kidney/morphological_subtyping/task.csv
  /home/kotpaz/scratch/tasks/custom/dhmc_luad/histologic_pattern/task.csv
  /home/kotpaz/scratch/tasks/custom/ebrains/diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/ebrains/diagnosis_group/task.csv
  /home/kotpaz/scratch/tasks/custom/ebrains/idh_status/task.csv
  /home/kotpaz/scratch/tasks/custom/imp/grade/task.csv
  /home/kotpaz/scratch/tasks/custom/mbc/treatment_response/task.csv
  /home/kotpaz/scratch/tasks/custom/panda/prostate_cancer_grade/task.csv
  /home/kotpaz/scratch/tasks/custom/imp_cervix/dysplasia_grading/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-BRCA/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-COAD/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-ESCA/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-SARC/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-TGCT/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-THYM/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA-UCEC/primary_diagnosis/task.csv
  /home/kotpaz/scratch/tasks/custom/TCGA/cancer_type_classification/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_lung/subtype/task.csv
  /home/kotpaz/scratch/tasks/custom/cptac_all/organ/task.csv
  /home/kotpaz/scratch/tasks/custom/mut-het-rcc/BAP1_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/mut-het-rcc/PBRM1_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/mut-het-rcc/SETD2_mutation/task.csv
  /home/kotpaz/scratch/tasks/custom/nadt/response/task.csv
  /home/kotpaz/scratch/tasks/custom/natbrca/lymphovascular_invasion/task.csv
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
DROPOUT="${DROPOUT:-0.25}"

# Feature directories to iterate (absolute paths)
# Update these with the actual per-WSI vector features ('.h5'/'hdf5' with dataset 'features')
FEATURE_DIRS=(
  /home/kotpaz/projects/rrg-msh/kotpaz/datasets
)
FEATURES_PARENT_DIR="linear_epoch_20"
FEAT_BASE_NAME="linear_epoch_20"

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
    feat_base="${FEAT_BASE_NAME}"
    if [[ ! -d "${fdir_noslash}" ]]; then
      echo "Warning: features dir not found: ${fdir_noslash}" >&2
    fi

    job_name="train_linear_${feat_base}_${dataset_name}_${task_name}"
    out_dir="results/${feat_base}/${dataset_name}/${task_name}/linear"

    echo "sbatch --job-name ${job_name} (features=${feat_base}, dataset=${dataset_name}, task=${task_name})"
    sbatch --job-name "${job_name}" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",FEATURES_SRC_DIR="${fdir_noslash}",FEATURES_PARENT_DIR="${FEATURES_PARENT_DIR}",DATASET="${dataset_name}",TASK="${task_name}",OUTPUT_DIR="${out_dir}",EPOCHS="${EPOCHS}",LR="${LR}",WEIGHT_DECAY="${WEIGHT_DECAY}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}",BALANCED_SAMPLING="${BALANCED_SAMPLING}",NORMALIZE="${NORMALIZE}",CSV_PATH="${csv_path}",NUM_FOLDS="${NUM_FOLDS}",DROPOUT="${DROPOUT}" \
      "${LAUNCH_SCRIPT}"
  done
done

echo "All submissions attempted."
