#input_path="../datasets/ai_coordination/dataset_train_subset_ab_cot.json"
input_path="../datasets/ai_coordination/dataset_testset_ab_cot.json"
output_path="../results/test_set_cot_ab/caa"
type="ab_cot"
# Define the arrays
multipliers=(1)
layers=(12)
python prompting_with_steering.py --behaviors "coordinate-other-ais" --layers $layers --multipliers $multipliers --type $type --model_size "7b" --input_path $input_path --output_path $output_path