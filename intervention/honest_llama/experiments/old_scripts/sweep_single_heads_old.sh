#!/bin/bash

# Define the array (output from the Python script)
# K_pairs=("12 16" "13 10" "26 22" "4 7" "4 6" "12 13" "2 31" "12 26" "12 29" "13 12" "10 30" "13 13")
#K_pairs=("13 0" "6 13" "29 29" "31 12" "12 16" "18 18" "15 6" "8 22" "15 5" "10 23" "31 14" "15 2" "10 22")
K_pairs=("13 0" "15 6" "30 0" "31 14" "15 5" "13 11" "16 28" "16 2" "10 22" "8 21" "14 1" "16 15" "31 28")
K_pairs=("13 0" "13 11" "31 14" "11 17" "31 28" "31 20" "31 22" "31 13" "30 16" "31 7" "31 5")
K_pairs=("13 0" "13 11" "31 14" "11 17" "31 13" "31 22")
K_pairs=("13 0" "13 11" "31 14" "22 15" "24 27" "3 3" "26 30" "3 10" "10 9" "3 14" "10 8")
K_pairs=("13 12")
K_pairs=("14 0" "14 10" "15 5" "15 6" "15 7" "16 6" "17 20")
K_pairs=("13 0" "15 5" "15 6" "15 7")
K_pairs=("13 0") #35 
#K_pairs=("13 11")
K_pairs=("15 5")
#K_pairs=("15 6")
K_pairs=("14 21")
K_pairs=("17 17")
#heads = [(0,26), (14,0), (14, 3), (15,5), (15,6), (17,12)]
K_pairs=("13 0" "0 26" "14 0" "14 3" "15 5" "15 6" "15 7" "17 12")
K_pairs=("13 0" "13 10") # "14 0" "14 3" "15 5" "15 6" "15 7" "17 12")
#K_pairs=("10 29" "12 19" "13 0" "14 3" "14 20" "14 21" "14 27")
model_name="openchat/openchat_3.5" #"meta-llama/Llama-2-7b-chat-hf" #"openchat/openchat_3.5"
input_path="../datasets/requirements_data/dataframe_open_chat_cot_moon_06022024_attentions_gt.json" #"../datasets/refusal/dataset_processed_attentions.json"
output_path="../intervention_results/requirements_data" #"../intervention_results/refusal_data"
dataset_name="requirements_data"
for alpha in 35 75; do
    for K in "${K_pairs[@]}"; do
        K1=$(echo $K | cut -d ' ' -f 1)
        K2=$(echo $K | cut -d ' ' -f 2)
        echo "alpha: $alpha K1: $K1 K2: $K2"
        python validate_2fold_moon_all_no_heads_analysis.py --layer $K1 --head $K2 --num_heads 1 --alpha $alpha --num_fold=1 --val_ratio=0.5 --model_name $model_name # --input_path $input_path --output_path $output_path --dataset $dataset_name
        echo
        echo
    done
done
