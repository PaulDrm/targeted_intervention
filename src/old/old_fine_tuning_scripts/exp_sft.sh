#!/bin/bash

CONFIG_FILE="experiments/requirements_data/config_valid_test_set_sft.json"

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
num_fold=$(echo $CONFIG | jq -r '.num_fold')
temperature=$(echo $CONFIG | jq -r '.temperature')
python scr_sft_baseline.py --model_name $model_name \
                           --input_path $input_path \
                           --output_path $output_path \
                           --num_fold $num_fold \
                           --temperature $temperature
