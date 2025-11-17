python scripts/compare_val_metrics.py \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v1/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v2/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v1/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v15/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/gigapath/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/midnight12k/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/dino_vit_small_p8_embeddings/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/cls/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/cls_l2/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/mean/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/mean_l2/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/linear/slide_features_feather/CAMELYON17 \
    /home/mila/k/kotpy/scratch/MIL-Lab/results/linear/slide_features_titan/CAMELYON17 \
    --names uni_v1 uni_v2 conch_v1 conch_v15 gigapath midnight12k backbone cls cls_l2 mean mean_l2 feather titan \
    --format latex --latex-file logs/cam17/cam_17.tex --only-intersection --latex-include-std --latex-include-embeddings --latex-compile --latex-keep-aux --latex-engine tectonic \
    --save-plots --plots-dir logs/cam17

# python scripts/compare_val_metrics.py \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v15/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/gigapath/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_0/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/midnight12k/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/dino_vit_small_p8_embeddings/CAMELYON17 \
#     --names uni_v1 uni_v2 conch_v1 conch_v15 virchow_v1 virchow_v2 hibou_b hibou_l gigapath phikon_v2 hoptimus_0 hoptimus_1 midnight12k backbone \
#     --format latex --latex-file logs/cam17/cam_17.tex --only-intersection --latex-include-std --latex-include-embeddings --latex-compile --latex-keep-aux --latex-engine tectonic \
#     --save-plots --plots-dir logs/cam17

# --exclude-methods dftd_afs meanmil transmil wikg

# python scripts/compare_val_metrics.py \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v15/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_b/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_l/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/gigapath/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_0/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/midnight12k/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/old_only_cam/clusters_50_100_500/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/old_only_cam/clusters_1000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/old_only_cam/clusters_1000_5000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/old_cam_tcga/clusters_1000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/old_cam_tcga/clusters_1000_5000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/old_cam_tcga/camelyon17_tcga_brca_clusters_1000_10000_trainable_temperature_separate_projection_heads_transformer_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_5000_10000_mask_prob_0.2_non_patch_included_no_cls_min_patch_ratio_0.05_overlap_0.75_transformer_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_5000_10000_mask_prob_0.2_no_cls_min_patch_ratio_0.05_overlap_0.75_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_5000_10000_mask_prob_0.2_no_cls_min_patch_ratio_0.05_overlap_0.75_transformer_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_5000_10000_mask_prob_0.2_no_cls_min_patch_ratio_0.05_overlap_0.75_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/mask_prob_0.6/camelyon17_tcga_brca_clusters_1000_5000_10000_separate_projection_heads_non_patch_included_no_cls_min_patch_ratio_0.5_overlap_0.5_transformer_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_5000_10000_trainable_temp_mask_prob_0.2_patch_overlap_0.5_min_patch_ratio_0.5_slide_overlap_0.5_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/dino_vit_small_p8_embeddings/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/feature_modeling_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/window_16_min_2_max_6_min_ratio_0.25_mask_prob_0.08_trainable_temp_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/window_16_min_2_max_6_min_ratio_0.25_mask_prob_0.2_trainable_temp_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/window_16_min_2_max_6_min_ratio_0.25_mask_prob_0.3_trainable_temp_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/window_16_min_2_max_6_min_ratio_0.25_mask_prob_0.4_trainable_temp_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/window_16_min_2_max_6_min_ratio_0.25_mask_prob_0.2_patch_overlap_0.5_trainable_temp_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/window_16_min_2_max_6_min_ratio_0.05_mask_prob_0.2_trainable_temp_transformer_separate_entries/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/window_16_min_2_max_6_min_ratio_0.05_mask_prob_0.2_patch_overlap_0.5_trainable_temp_transformer_separate_entries/CAMELYON17 \
#     --names uni_v1 uni_v2 conch_v1 conch_v15 virchow_v1 virchow_v2 hibou_b hibou_l gigapath phikon_v2 hoptimus_0 hoptimus_1 midnight12k 50_100_500 1000_10000 1000_5000_10000 tcga_and_cam_1000_10000 tcga_and_cam_1000_5000_10000 tcga_and_cam_1000_10000_trainable_temp 1000_5000_10000_average_non_patch_inc_min_ratio_0.05_overlap_0.75 like_prev_but_separate 1000_5000_10000_average_min_ratio_0.05_overlap_0.75 like_prev_but_separate prob_0.6 patch_overlap_0.5 backbone feature_modeling window_16_mask_0.08 window_16_mask_0.2 window_16_mask_0.3 window_16_mask_0.4 window_16_mask_0.2_patch_overlap_0.5 window_16_mask_0.2_min_ratio_0.05 window_16_mask_0.2_min_ratio_0.05_overlap_0.5 \
#     --format latex --latex-file logs/cam17/cam_17.tex --only-intersection --latex-include-std --latex-compile --latex-keep-aux --latex-engine tectonic \
#     --save-plots --plots-dir logs/cam17

# python scripts/compare_val_metrics.py \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v2/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v15/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow2/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_b/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_l/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/gigapath/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v2/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_0/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/midnight12k/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_5000_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_1000_5000_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_separate_projection_heads_transformer_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_trainable_temperature_separate_projection_heads_transformer_average_overlaps/TCGA_BRCA \
#     --names uni_v1 uni_v2 conch_v1 conch_v15 virchow_v1 virchow_v2 hibou_b hibou_l gigapath phikon_v2 hoptimus_0 hoptimus_1 midnight12k 10000_ 5000_10000 1000_5000_10000 1000_10000 1000_10000_trainable_temp \
#     --format latex --latex-file logs/tcga_brca/tcga_brca.tex --only-intersection --latex-include-std --latex-compile --latex-keep-aux --latex-engine tectonic \
#     --save-plots --plots-dir logs/tcga_brca

# python scripts/compare_val_metrics.py \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v2/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v15/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow2/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_b/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_l/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/dino_vit_small_p8_embeddings/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/gigapath/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v2/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/resnet50/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_0/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/midnight12k/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_1000_5000_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_1000_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_5000_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     --names uni_v1 uni_v2 conch_v1 conch_v15 virchow_v1 virchow_v2 hibou_b hibou_l backbone gigapath phikon_v1 phikon_v2 resnet50 hoptimus_0 hoptimus_1 midnight12k 1000_5000_10000 1000_10000 5000_10000 clusters_10000 \
#     --save-plots --plots-dir logs/compare_plots_tcga_brca --only-intersection

# python scripts/compare_val_metrics.py \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v15/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_b/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_l/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/dino_vit_small_p8_embeddings/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/gigapath/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/resnet50/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_0/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/midnight12k/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_1000_5000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_5000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_10000_steps_20K_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_1000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_trainable_temperature_separate_projection_heads_transformer_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_trainable_temperature_separate_projection_heads_concat_codebook_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_trainable_temperature_separate_projection_heads_concat_projection_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_separate_projection_heads_transformer_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/old_only_cam/clusters_50_100_500/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/old_only_cam/clusters_1000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     --names uni_v1 uni_v2 conch_v1 conch_v15 virchow_v1 virchow_v2 hibou_b hibou_l backbone gigapath phikon_v1 phikon_v2 resnet50 hoptimus_0 hoptimus_1 midnight12k 1000_50000_10000 5000_10000 10000c 1000_10000 temp_separate_transformer_1000_10000 temp_separate_concat_codebook_1000_10000 temp_separate_concat_projection_1000_10000 separate_projection_heads_transformer_1000_10000 old_only_cam_50_100_500 old_only_cam_1000_10000 \
#     --save-plots --plots-dir logs/compare_plots_camelyon17 --only-intersection

# python scripts/compare_val_metrics.py \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v2/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v15/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow2/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_b/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_l/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/dino_vit_small_p8_embeddings/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/gigapath/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v2/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/resnet50/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_0/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_1/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/midnight12k/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_1000_5000_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_5000_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_1000_10000_steps_20K_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_trainable_temperature_separate_projection_heads_transformer_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_trainable_temperature_separate_projection_heads_concat_codebook_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_trainable_temperature_separate_projection_heads_concat_projection_average_overlaps/TCGA_BRCA \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/camelyon17_tcga_brca_clusters_1000_10000_separate_projection_heads_transformer_average_overlaps/TCGA_BRCA \
#     --names uni_v1 uni_v2 conch_v1 conch_v15 virchow_v1 virchow_v2 hibou_b hibou_l backbone gigapath phikon_v1 phikon_v2 resnet50 hoptimus_0 hoptimus_1 midnight12k 1000_5000_10000 5000_10000 10000c 1000_10000 temp_separate_transformer_1000_10000 temp_separate_concat_codebook_1000_10000 temp_separate_concat_projection_1000_10000 separate_projection_heads_transformer_1000_10000 \
#     --save-plots --plots-dir logs/compare_plots_tcga_brca --only-intersection

# python scripts/compare_val_metrics.py \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/uni_v2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/conch_v15/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/virchow2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_b/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hibou_l/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/dino_vit_small_p8_embeddings/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/gigapath/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/phikon_v2/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/resnet50/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_0/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/hoptimus_1/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/midnight12k/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/clusters_1000_5000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     /home/mila/k/kotpy/scratch/MIL-Lab/results/old_only_cam/clusters_1000_5000_10000_steps_20K_average_overlaps/CAMELYON17 \
#     --names uni_v1 uni_v2 conch_v1 conch_v15 virchow_v1 virchow_v2 hibou_b hibou_l backbone gigapath phikon_v1 phikon_v2 resnet50 hoptimus_0 hoptimus_1 midnight12k 1000_5000_10000 old_only_cam_1000_5000_10000 \
#     --save-plots --plots-dir logs/compare_plots_camelyon17 --only-intersection
