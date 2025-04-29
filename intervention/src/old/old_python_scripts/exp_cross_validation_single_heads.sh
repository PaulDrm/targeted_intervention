#!/bin/bash

# Default configuration file
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
# Get the K_pairs as a single JSON array of arrays
K_pairs=$(echo $CONFIG | jq -c '.K_pairs')
alphas=$(echo $CONFIG | jq -r '.alphas[]')
num_head=1 #$(echo $CONFIG | jq -r '.num_heads[]')
use_center_of_mass=$(echo $CONFIG | jq -r '.use_center_of_mass')
val_ratio=$(echo $CONFIG | jq -r '.val_ratio')
num_fold=$(echo $CONFIG | jq -r '.num_fold')
temperature=$(echo $CONFIG | jq -r '.temperature')
echo $K_pairs

# Use jq to iterate over the pairs within the JSON array

    first_element=$(echo $pair | jq -r '.[0]')


for alpha in $alphas; do
  for pair in $(echo $K_pairs | jq -c '.[]'); do
    # Extract the first element of each pair
    echo $pair
    python scr_validate_multiple_fold_multiple_heads.py --num_heads $num_head \
                                                       --alpha $alpha \
                                                       --num_fold=$num_fold \
                                                       --model_name $model_name  \
                                                       --input_path $input_path \
                                                       --output_path $output_path \
                                                       --dataset $dataset_name \
                                                       --add_or_subtract true \
                                                       --use_center_of_mass $use_center_of_mass \
                                                       --list_of_heads "[$pair]" \
                                                       --temperature $temperature
    echo
  done
done
