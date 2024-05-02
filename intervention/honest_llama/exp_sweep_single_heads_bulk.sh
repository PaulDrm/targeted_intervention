#!/bin/bash

model_name="meta-llama/Llama-2-7b-chat-hf" #"openchat/openchat_3.5" #"meta-llama/Llama-2-7b-chat-hf" #"openchat/openchat_3.5"
input_path="../datasets/ai_coordination/dataset_241_selection_attentions.json" #"../datasets/ai_coordination/dataset_processed_attentions.json"  #"../datasets/refusal/dataset_processed_attentions.json" #"../datasets/requirements_data/dataframe_open_chat_cot_moon_06022024_attentions_gt.json" #"../datasets/refusal/dataset_processed_attentions.json"
output_path="../intervention_results/ai_coordination/bulk_results_single_example" #"../intervention_results/ai_coordination/bulk_results" #"../intervention_results/refusal_data/bulk_results" #"../intervention_results/requirements_data" #"../intervention_results/refusal_data"
dataset_name="ai_coordination" #"refusal" #"requirements_data"
num_layers=24
num_heads=32
valratio=0
prompttype="ab_cot"
for alpha in 35 75; do
    for ((layer = 0; layer < num_layers; layer++)); do
        for ((head = 0; head < num_heads; head++)); do
            echo "alpha: $alpha Layer: $layer Head: $head"
            python validate_2fold_moon_all_no_heads_analysis_new.py --layer $layer --head $head --num_heads 1 --alpha $alpha --num_fold=1 --val_ratio=$valratio --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --prompt_type $prompttype
            echo
        done
    done
done
