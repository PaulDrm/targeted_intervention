#!/bin/bash

CONFIG_FILE="$1"
# Read and parse the configuration file
CONFIG=$(jq '.' "$CONFIG_FILE")

# Load paths from config.json
input_path=$(echo $CONFIG | jq -r '.input_path')
output_path=$(echo $CONFIG | jq -r '.output_path')
test_mode=$(echo $CONFIG | jq -r '.test_mode')
echo "Input path: $input_path"
echo "Output path: $output_path"

# Run the Python script with the paths
python scr_gpt4_truthful_scoring.py --input_path "$input_path"\
                                     --output_path "$output_path" \
                                     --test_mode $test_mode
