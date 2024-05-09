#!/bin/bash

#CONFIG=$(jq '.' experiments/ai_coordination/config_example_294_bulk.json)
CONFIG=$(jq '.' experiments/ai_coordination/config_example_294_bulk_com.json)
CONFIG=$(jq '.' experiments/ai_coordination/config_example_304_bulk_com.json)
CONFIG=$(jq '.' experiments/ai_coordination/config_example_307_bulk_com.json)
CONFIG=$(jq '.' experiments/ai_coordination/config_example_307_train_sweep_com_no_std.json)
model_name=$(echo $CONFIG | jq -r '.model_name')
input_path=$(echo $CONFIG | jq -r '.input_path')
output_path=$(echo $CONFIG | jq -r '.output_path')
dataset_name=$(echo $CONFIG | jq -r '.dataset_name')
use_center_of_mass=$(echo $CONFIG | jq -r '.use_center_of_mass')
add_proj_val_std=$(echo $CONFIG | jq -r '.add_proj_val_std')

dataset_name="ai_coordination" #"refusal" #"requirements_data"
val_ratio=0
prompt_type="ab_cot"
#--layer $layer --head $head --num_heads 1 --alpha $alpha --num_fold=1 --val_ratio=$valratio --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --prompt_type $prompttype
for alpha in 35 75 150; do
    echo "alpha: $alpha"
    python scr_validate_sweep_heads_probing.py --alpha $alpha --val_ratio $val_ratio --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --prompt_type $prompt_type --use_center_of_mass $use_center_of_mass --add_proj_val_std $add_proj_val_std
    echo
    echo
    done
done
