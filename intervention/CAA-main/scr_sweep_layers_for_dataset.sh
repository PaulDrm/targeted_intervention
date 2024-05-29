#input_path="../datasets/ai_coordination/dataset_train_subset_ab_cot.json"
input_path="../datasets/ai_coordination/dataset_val_subset_100_ab_cot.json"
python prompting_with_steering.py --behaviors "coordinate-other-ais" --layers $(seq 0 31) --multipliers 1 2 5 --type "ab_cot" --model_size "7b" --input_path $input_path --output_path "../results/CAA/ab_cot_val"
