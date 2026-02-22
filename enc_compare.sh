source /home/kotpaz/envs/atlas_path/bin/activate

# python scripts/compare_encoders_across_tasks.py --encoders prism madeleine titan stage_1 stage_2 --methods linear --places 2
python scripts/compare_encoders_across_tasks.py --encoders prism madeleine titan stage_1 stage_2 --methods linear --places 2 --include-tasks scripts/include_tasks.txt

# python scripts/compare_encoders_across_tasks.py --encoders titan prism feather madeleine linear_epoch_20 --methods linear
# python scripts/compare_encoders_across_tasks.py --encoders titan prism feather madeleine linear_epoch_20 --methods knn
# python scripts/compare_encoders_across_tasks.py --encoders titan prism feather madeleine linear_epoch_20 --methods logreg
# python scripts/compare_encoders_across_tasks.py --encoders titan prism feather madeleine case_agg_linear_epoch_6 --methods linear
# python scripts/compare_encoders_across_tasks.py --encoders titan prism feather madeleine case_agg_linear_epoch_6 --methods linear --filter-winner case_agg_linear_epoch_6
