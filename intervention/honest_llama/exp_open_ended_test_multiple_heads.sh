#!/bin/bash

#CONFIG=$(jq '.' experiments/ai_coordination/config_targeted_test_single_head.json)

#CONFIG=$(jq '.' experiments/ai_coordination/config_targeted_test_single_heads_294_train_com.json)
#CONFIG=$(jq '.' experiments/ai_coordination/config_train_set_tests_intervention_294_multi_heads.json)
#CONFIG=$(jq '.' experiments/ai_coordination/config_open_ended_test_multiple_heads_ab_cot_294_train_extract.json)

#CONFIG=$(jq '.' experiments/ai_coordination/config_open_ended_test_multiple_heads_ab_cot_294_304_train_extract.json)

#CONFIG=$(jq '.' experiments/ai_coordination/config_open_ended_test_multiple_heads_ab_cot_294_304_307_train_extract.json)

CONFIG_FILE="experiments/ai_coordination/config_open_ended_test_multiple_heads_ab_cot_294_304_307_test.json"

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
# Get the K_pairs as a single JSON array of arrays
K_pairs=$(echo $CONFIG | jq '.K_pairs')
alphas=$(echo $CONFIG | jq -r '.alphas[]')
use_center_of_mass=$(echo $CONFIG | jq -r '.use_center_of_mass')

dataset_name="ai_coordination" #"refusal" #"requirements_data"
prompt_type="open_ended" #"ab_cot" # 
val_ratio=0 # 0.5
# K_pairs=("11 22" "12 19") #("14 18")
for alpha in $alphas; do
  #for K in $K_pairs; do
  python scr_validate_intervention_for_dataset_multiple_heads.py --num_heads 1 --alpha $alpha --num_fold=1 --val_ratio=$val_ratio --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --add_or_subtract true --test_set_input_path $test_data_path --prompt_type $prompt_type --use_center_of_mass $use_center_of_mass --list_of_heads "$K_pairs"
  echo
done
