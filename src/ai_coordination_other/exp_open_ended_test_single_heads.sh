#!/bin/bash


#CONFIG=$(jq '.' experiments/ai_coordination/config_open_ended_test_single_heads_ab_cot_294.json)
#CONFIG=$(jq '.' experiments/ai_coordination/config_open_ended_test_single_heads_ab_cot_294_train_extract.json)

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
K_pairs=$(echo $CONFIG | jq -c '.K_pairs[]')  # This fetches arrays of pairs
alphas=$(echo $CONFIG | jq -r '.alphas[]')
use_center_of_mass=$(echo $CONFIG | jq -r '.use_center_of_mass')
temperature=$(echo $CONFIG | jq -r '.temperature')



dataset_name="ai_coordination" #"refusal" #"requirements_data"
prompt_type="open_ended"
val_ratio=0
# K_pairs=("11 22" "12 19") #("14 18")
for alpha in $alphas; do
  #for K in $K_pairs; do
  for pair in $K_pairs; do
    layer=$(echo $pair | jq -r '.[0]')  # Parse first element
    head=$(echo $pair | jq -r '.[1]')   # Parse second element
    echo "alpha: $alpha Layer: $layer Head: $head"
    #python scr_validate_intervention_for_dataset.py --layer $layer --head $head --num_heads 1 --alpha $alpha --num_fold=1 --val_ratio=$val_ratio --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --add_or_subtract true --test_set_input_path $test_data_path --prompt_type $prompt_type --use_center_of_mass $use_center_of_mass
    python scr_validate_intervention_for_dataset_improved.py --layer $layer \
                                                             --head $head \
                                                             --num_heads 1 \
                                                             --alpha $alpha \
                                                             --num_fold=1 \
                                                             --val_ratio=$val_ratio \
                                                             --model_name $model_name \
                                                             --input_path $input_path \
                                                             --output_path $output_path \
                                                             --dataset $dataset_name \
                                                             --add_or_subtract true \
                                                             --test_set_input_path $test_data_path \
                                                             --prompt_type $prompt_type \
                                                             --use_center_of_mass $use_center_of_mass \
                                                             --temperature $temperature
    echo
    done
done
