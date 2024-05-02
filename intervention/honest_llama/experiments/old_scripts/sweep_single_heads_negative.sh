#!/bin/bash

model_name="meta-llama/Llama-2-7b-chat-hf" #"openchat/openchat_3.5" #"meta-llama/Llama-2-7b-chat-hf" #"openchat/openchat_3.5"
input_path="../datasets/refusal/dataset_processed_attentions.json" #"../datasets/requirements_data/dataframe_open_chat_cot_moon_06022024_attentions_gt.json" #"../datasets/refusal/dataset_processed_attentions.json"
output_path="../intervention_results/refusal_data/negative_results" #"../intervention_results/requirements_data" #"../intervention_results/refusal_data"
dataset_name="refusal" #"requirements_data"

#K_pairs=("12 6" "14 27" "12 0" "12 1")
K_pairs=("12 6")
for alpha in 150 200 ; do
    for K in "${K_pairs[@]}"; do
        layer=$(echo $K | cut -d ' ' -f 1)
        head=$(echo $K | cut -d ' ' -f 2)
        echo "alpha: $alpha Layer: $layer Head: $head"
        python validate_2fold_moon_all_no_heads_analysis_new.py --layer $layer --head $head --num_heads 1 --alpha $alpha --num_fold=1 --val_ratio=0.5 --model_name $model_name  --input_path $input_path --output_path $output_path --dataset $dataset_name --add_or_subtract false
        echo
    done
done
