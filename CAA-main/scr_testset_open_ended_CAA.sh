#input_path="../datasets/ai_coordination/dataset_train_subset_ab_cot.json"
input_path="../datasets/ai_coordination/dataset_testset_ab_cot.json"
output_path="../results/test_set_open_ended/caa"
type="open_ended"
# Define the arrays
multipliers=(1 2 5)
layers=(12 13 15)
python prompting_with_steering.py --behaviors "coordinate-other-ais" --layers $layers --multipliers $multipliers --type $type --model_size "7b" --output_path $output_path --input_path $input_path