#!/bin/bash

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
num_fold=$(echo $CONFIG | jq -r '.num_fold')
temperature=$(echo $CONFIG | jq -r '.temperature')
consistency_factors=$(echo $CONFIG | jq -c '.consistency_factors')
consistency_factor=$(echo $CONFIG | jq -r '.consistency_factor')
prompt_type=$(jq -r '.prompt_type // "ab_cot"'        "$CONFIG_FILE")
seeds=$(echo $CONFIG | jq -c '.seeds') # Extract individual seed(s)
limit=$(echo $CONFIG | jq -r '.limit')
test_set=$(echo $CONFIG | jq -r '.test_set')
input_path_configuration=$(echo $CONFIG | jq -r '.input_path_configuration')
num_mc=$(echo $CONFIG | jq -r '.num_mc')
alpha_sweep=$(echo $CONFIG | jq -r '.alpha_sweep')
val_ratio=$(echo $CONFIG | jq -r '.val_ratio')
temperatures=$(echo $CONFIG | jq -c '.temperatures')
upper_bound=$(echo $CONFIG | jq -r '.upper_bound')
lower_bound=$(echo $CONFIG | jq -r '.lower_bound')
max_new_tokens=$(jq -r '.max_new_tokens // "600"'       "$CONFIG_FILE")
batch_size=$(jq -r '.batch_size // "2"' "$CONFIG_FILE")
scoring=$(jq -r '.scoring // "false"' "$CONFIG_FILE")

echo $val_ratio
echo $seeds
echo $consistency_factors
echo $num_fold
echo $scoring

# Define relative path to Python script
PYTHON_SCRIPT="../src/intervention/scr_evaluate_configurations_general.py"

python "$PYTHON_SCRIPT" \
          --num_fold=$num_fold \
          --model_name $model_name  \
          --input_path $input_path \
          --output_path $output_path \
          --dataset $dataset_name \
          --use_center_of_mass $use_center_of_mass \
          --temperature $temperature \
          --consistency_factors $consistency_factors \
          --consistency_factor $consistency_factor \
          --seeds $seeds \
          --limit $limit \
          --test_set $test_set \
          --input_path_configuration $input_path_configuration \
          --num_mc $num_mc \
          --alpha_sweep $alpha_sweep \
          --val_ratio $val_ratio \
          --temperatures $temperatures \
          --upper_bound $upper_bound \
          --lower_bound $lower_bound \
          --prompt_type $prompt_type \
          --max_new_tokens $max_new_tokens \
          --batch_size $batch_size \
          --scoring $scoring 
#echo done
