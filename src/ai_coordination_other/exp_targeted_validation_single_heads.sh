#!/bin/bash

#CONFIG=$(jq '.' experiments/ai_coordination/config_targeted_test_single_head.json)

#CONFIG=$(jq '.' experiments/ai_coordination/config_targeted_test_single_heads_294_train_com.json)
#CONFIG=$(jq '.' experiments/ai_coordination/config_train_sets_test_single_heads_294_304_train_com.json)

#CONFIG=$(jq '.' experiments/ai_coordination/config_train_set_tests_intervention_304.json)
#CONFIG=$(jq '.' experiments/ai_coordination/config_example_307_extra_heads_com.json)

#CONFIG=$(jq '.' experiments/ai_coordination/config_train_subset_tests_intervention_373.json)
#CONFIG=$(jq '.' experiments/ai_coordination/config_example_307_on_373_single_example.json)

CONFIG_FILE="experiments/ai_coordination/config_train_set_tests_intervention_294_304_307_new_multi_heads.json"

# Check if a configuration file is provided
if [ -n "$1" ]; then
  CONFIG_FILE="$1"
else
  echo "Usage: $0 <config_file>"
  echo "Using default configuration file: $CONFIG_FILE"
fi

# Read and parse the configuration file
CONFIG=$(jq '.' "$CONFIG_FILE")

model_name=$(echo $CONFIG | jq -r '.model_name')
input_path=$(echo $CONFIG | jq -r '.input_path')
output_path=$(echo $CONFIG | jq -r '.output_path')
dataset_name=$(echo $CONFIG | jq -r '.dataset_name')
test_data_path=$(echo $CONFIG | jq -r '.test_data_path')
K_pairs=$(echo $CONFIG | jq -c '.K_pairs[]')  # This fetches arrays of pairs
alphas=$(echo $CONFIG | jq -r '.alphas[]')
use_center_of_mass=$(echo $CONFIG | jq -r '.use_center_of_mass')
dataset_name=$(echo $CONFIG | jq -r '.dataset_name') #"refusal" #"requirements_data"
temperature=$(echo $CONFIG | jq -r '.temperature')
limit=$(echo $CONFIG | jq -r '.limit')
prompt_type="ab_cot" # "open_ended"

val_ratio=0 # 0.5

# K_pairs=("11 22" "12 19") #("14 18")
for alpha in $alphas; do
  #for K in $K_pairs; do
  for pair in $K_pairs; do
      layer=$(echo $pair | jq -r '.[0]')  # Parse first element
      head=$(echo $pair | jq -r '.[1]')   # Parse second element
      echo "alpha: $alpha Layer: $layer Head: $head"
      # scr_validate_intervention_for_dataset.py --layer $layer \
      python scr_validate_intervention_for_dataset_improved.py --layer $layer \
                                                      --head $head \
                                                      --num_heads 1 \
                                                      --alpha $alpha \
                                                      --num_fold=1 \
                                                      --val_ratio=$val_ratio \
                                                      --model_name $model_name \
                                                      --input_path $input_path \
                                                      --output_path $output_path \
                                                      --dataset $dataset_name \
                                                      --add_or_subtract true \
                                                      --test_set_input_path $test_data_path \
                                                      --prompt_type $prompt_type \
                                                      --use_center_of_mass $use_center_of_mass \
                                                      --temperature $temperature \
                                                      --limit $limit

    echo
    done
done
