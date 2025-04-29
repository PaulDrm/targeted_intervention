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
consistency_factor=$(echo $CONFIG | jq -r '.consistency_factor')
seeds=$(echo $CONFIG | jq -r '.seeds[]') # Extract individual seed(s)

python scr_algorithm_split_and_optimize_heads_iteratively.py  --num_fold=$num_fold \
                                                  --model_name $model_name  \
                                                  --input_path $input_path \
                                                  --output_path $output_path \
                                                  --dataset $dataset_name \
                                                  --add_or_subtract true \
                                                  --use_center_of_mass $use_center_of_mass \
                                                  --temperature $temperature \
                                                  --consistency_factor $consistency_factor \
                                                  --seeds $seeds

echo

done
