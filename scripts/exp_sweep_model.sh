#!/bin/bash

# Check if a configuration file is provided, otherwise use a default
DEFAULT_CONFIG_FILE="config_sweep.json"
if [ -n "$1" ]; then
  CONFIG_FILE="$1"
else
  echo "Usage: $0 <config_file>"
  echo "Using default configuration file: $DEFAULT_CONFIG_FILE"
  CONFIG_FILE="$DEFAULT_CONFIG_FILE"
fi

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Configuration file '$CONFIG_FILE' not found."
  exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "Error: jq is not installed. Please install jq to parse the config file."
    exit 1
fi

# Read the configuration file using jq
CONFIG_CONTENT=$(cat "$CONFIG_FILE") # Read entire file once

# --- Extract parameters from JSON configuration ---

# Required parameters (Python script will error if these are missing/null from JSON)
initial_dataset_path=$(echo "$CONFIG_CONTENT" | jq -r '.initial_dataset_path')
evaluation_dataset_path=$(echo "$CONFIG_CONTENT" | jq -r '.evaluation_dataset_path')
layers_to_sweep=$(echo "$CONFIG_CONTENT" | jq -r '.layers_to_sweep')

# Optional parameters with direct defaults (these will always be passed to Python)
# Their defaults in jq match the Python script's internal defaults.
model_name=$(echo "$CONFIG_CONTENT" | jq -r '.model_name // "meta-llama/Llama-2-7b-chat-hf"')
heads_to_sweep_per_layer=$(echo "$CONFIG_CONTENT" | jq -r '.heads_to_sweep_per_layer // ""') # Python handles "" as "all heads"
max_new_tokens=$(echo "$CONFIG_CONTENT" | jq -r '.max_new_tokens // "600"')
temperature=$(echo "$CONFIG_CONTENT" | jq -r '.temperature // "0.1"') # Final eval temp
seed=$(echo "$CONFIG_CONTENT" | jq -r '.seed // "42"')
output_dir=$(echo "$CONFIG_CONTENT" | jq -r '.output_dir // "./results"') # Matches Python default
id_column=$(echo "$CONFIG_CONTENT" | jq -r '.id_column // "data_id"')
batch_size=$(echo "$CONFIG_CONTENT" | jq -r '.batch_size // "2"')
consistency_factor=$(echo "$CONFIG_CONTENT" | jq -r '.consistency_factor // "1"')
dataset_name_for_processing=$(echo "$CONFIG_CONTENT" | jq -r '.dataset_name_for_processing // "ai_coordination"')
prompt_type_for_processing=$(echo "$CONFIG_CONTENT" | jq -r '.prompt_type_for_processing // "ab_cot"')
alpha=$(echo "$CONFIG_CONTENT" | jq -r '.alpha // "none"')
alphas=$(echo "$CONFIG_CONTENT" | jq -r '.alphas // "none"')
# Boolean flags (defaults in jq match Python script's argparse defaults)
use_center_of_mass_str=$(echo "$CONFIG_CONTENT" | jq -r '.use_center_of_mass // "true"') # Python script default is True
use_random_dir_str=$(echo "$CONFIG_CONTENT" | jq -r '.use_random_dir // "false"')   # Python script default is False

# --- Basic validation for truly required fields (if jq returns 'null' string or empty) ---
if [ -z "$initial_dataset_path" ] || [ "$initial_dataset_path" == "null" ]; then
  echo "Error: 'initial_dataset_path' is required and was not found or was null in $CONFIG_FILE."
  exit 1
fi
if [ -z "$evaluation_dataset_path" ] || [ "$evaluation_dataset_path" == "null" ]; then
  echo "Error: 'evaluation_dataset_path' is required and was not found or was null in $CONFIG_FILE."
  exit 1
fi
if [ -z "$layers_to_sweep" ] || [ "$layers_to_sweep" == "null" ]; then
  echo "Error: 'layers_to_sweep' is required and was not found or was null in $CONFIG_FILE."
  exit 1
fi

# --- Construct the Python script command ---
CMD_ARGS=""

# Append required and direct-default optional arguments
CMD_ARGS+=" --initial_dataset_path \"$initial_dataset_path\""
CMD_ARGS+=" --evaluation_dataset_path \"$evaluation_dataset_path\""
CMD_ARGS+=" --layers_to_sweep \"$layers_to_sweep\"" # Quote for comma-separated list
CMD_ARGS+=" --model_name \"$model_name\""
CMD_ARGS+=" --heads_to_sweep_per_layer \"$heads_to_sweep_per_layer\"" # Quote if it can contain spaces/commas
CMD_ARGS+=" --alpha $alpha"
CMD_ARGS+=" --max_new_tokens $max_new_tokens"
CMD_ARGS+=" --temperature $temperature"
CMD_ARGS+=" --seed $seed"
CMD_ARGS+=" --output_dir \"$output_dir\""
CMD_ARGS+=" --batch_size $batch_size"
CMD_ARGS+=" --consistency_factor $consistency_factor"
CMD_ARGS+=" --dataset_name \"$dataset_name_for_processing\""
CMD_ARGS+=" --prompt_type \"$prompt_type_for_processing\""
CMD_ARGS+=" --alphas $alphas"

# Append conditional arguments (those using "none" placeholder)
if [ "$cache_dir" != "none" ]; then
  CMD_ARGS+=" --cache_dir \"$cache_dir\""
fi

# Append boolean flags
if [ "$use_center_of_mass_str" == "true" ]; then
  CMD_ARGS+=" --use_center_of_mass"
fi
if [ "$use_random_dir_str" == "true" ]; then
  CMD_ARGS+=" --use_random_dir"
fi

# Define the path to your Python script
# Adjust this path if run_head_sweep.py is located elsewhere
PYTHON_SCRIPT="../src/intervention/scr_sweep.py"

# --- Execute the Python script ---
echo "Running Python script: $PYTHON_SCRIPT with arguments:"
echo "$CMD_ARGS"
echo "---"

eval "python $PYTHON_SCRIPT $CMD_ARGS"

echo "---"
echo "Script execution finished."