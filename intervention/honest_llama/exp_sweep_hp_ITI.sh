#!/bin/bash

CONFIG_FILE="config_train_subset_tests_intervention_ITI.json"


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
use_center_of_mass=$(echo $CONFIG | jq -r '.use_center_of_mass')
alphas=$(echo $CONFIG | jq -r '.alphas[]')
Ks=$(echo $CONFIG | jq -r '.Ks[]')
val_ratio=$(echo $CONFIG | jq -r '.val_ratio')
prompt_type="ab_cot"

# --num_heads 1 --alpha $alpha --num_fold=1 --val_ratio=$val_ratio --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --add_or_subtract true --test_set_input_path $test_data_path --prompt_type $prompt_type --use_center_of_mass $use_center_of_mass --list_of_heads "$K_pairs"
#5 15 20 35
for alpha in $alphas; do
    for K in $Ks; do
        echo "alpha: $alpha K: $K"
        python scr_validate_intervention_for_dataset_multiple_heads.py  --num_heads $K --alpha $alpha --num_fold=1 --val_ratio=$val_ratio --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --add_or_subtract true --test_set_input_path $test_data_path --prompt_type $prompt_type --use_center_of_mass $use_center_of_mass
        echo
        echo
    done
done
