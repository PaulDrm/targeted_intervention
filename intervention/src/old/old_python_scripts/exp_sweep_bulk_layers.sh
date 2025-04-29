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
use_center_of_mass=$(echo $CONFIG | jq -r '.use_center_of_mass')
temperature=$(echo $CONFIG | jq -r '.temperature')
val_ratio=$(echo $CONFIG | jq -r '.val_ratio')

echo $val_ratio
#val_ratio=0

#add_proj_val_std=$(echo $CONFIG | jq -r '.add_proj_val_std')

#dataset_name="ai_coordination" #"refusal" #"requirements_data"

prompt_type="ab_cot"
#--layer $layer --head $head --num_heads 1 --alpha $alpha --num_fold=1 --val_ratio=$valratio --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --prompt_type $prompttype
#--add_proj_val_std $add_proj_val_std
for alpha in 1; do
    echo "alpha: $alpha"
    #python scr_validate_sweep_heads_probing_batch.py --alpha $alpha --val_ratio $val_ratio --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --prompt_type $prompt_type --use_center_of_mass $use_center_of_mass --temperature $temperature
    #python scr_validate_sweep_heads_probing.py --alpha $alpha \
    
    python scr_validate_sweep_layers_probing_improved.py --alpha $alpha \
                                               --val_ratio $val_ratio \
                                               --model_name $model_name \
                                               --input_path $input_path \
                                               --output_path $output_path \
                                               --dataset $dataset_name \
                                               --prompt_type $prompt_type \
                                               --use_center_of_mass $use_center_of_mass \
                                               --temperature $temperature
    
    echo
    done
done
