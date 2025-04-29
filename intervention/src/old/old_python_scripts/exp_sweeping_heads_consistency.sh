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
# Get the K_pairs as a single JSON array of arrays
use_center_of_mass=$(echo $CONFIG | jq -r '.use_center_of_mass')
val_ratio=$(echo $CONFIG | jq -r '.val_ratio')
num_fold=$(echo $CONFIG | jq -r '.num_fold')
temperature=$(echo $CONFIG | jq -r '.temperature')
consistency_factors=$(echo $CONFIG | jq -c '.consistency_factors')
seeds=$(echo $CONFIG | jq -c '.seeds') # Extract individual seed(s)
num_mc=$(echo $CONFIG | jq -r '.num_mc')

echo $seeds
echo $consistency_factors
echo $val_ratio

python scr_evaluate_configurations.py --num_fold=$num_fold \
                                      --model_name $model_name  \
                                      --input_path $input_path \
                                      --output_path $output_path \
                                      --dataset $dataset_name \
                                      --use_center_of_mass $use_center_of_mass \
                                      --temperature $temperature \
                                      --consistency_factors $consistency_factors \
                                      --val_ratio $val_ratio \
                                      --seeds $seeds
echo done
