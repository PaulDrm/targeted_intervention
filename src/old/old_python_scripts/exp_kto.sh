#!/bin/bash

CONFIG_FILE="experiments/requirements_data/config_valid_test_set_sft.json"


# Check if a configuration file is provided
if [ -n "$1" ]; then
  CONFIG_FILE="$1"
else
  echo "Usage: $0 <config_file>"
  echo "Using default configuration file: $CONFIG_FILE"
fi

# Read and parse the configuration file
CONFIG=$(jq '.' "$CONFIG_FILE")

echo "Running script"
model_name=$(echo $CONFIG | jq -r '.model_name')
input_path=$(echo $CONFIG | jq -r '.input_path')
output_path=$(echo $CONFIG | jq -r '.output_path')
num_fold=$(echo $CONFIG | jq -r '.num_fold')
temperature=$(echo $CONFIG | jq -r '.temperature')
# Reading each field from JSON using `jq`
#MODEL_NAME_OR_PATH=$(jq -r '.model_name_or_path' $CONFIG_FILE)
PER_DEVICE_TRAIN_BATCH_SIZE=$(jq -r '.per_device_train_batch_size' $CONFIG_FILE)
PER_DEVICE_EVAL_BATCH_SIZE=$(jq -r '.per_device_eval_batch_size' $CONFIG_FILE)
NUM_TRAIN_EPOCHS=$(jq -r '.num_train_epochs' $CONFIG_FILE)
#LEARNING_RATE=$(jq -r '.learning_rate' $CONFIG_FILE)
LEARNING_RATES=$(echo $CONFIG | jq -r '.learning_rate[]')
LR_SCHEDULER_TYPE=$(jq -r '.lr_scheduler_type' $CONFIG_FILE)
GRADIENT_ACCUMULATION_STEPS=$(jq -r '.gradient_accumulation_steps' $CONFIG_FILE)
LOGGING_STEPS=$(jq -r '.logging_steps' $CONFIG_FILE)
OUTPUT_DIR=$(jq -r '.output_dir' $CONFIG_FILE)
WARMUP_RATIO=$(jq -r '.warmup_ratio' $CONFIG_FILE)
REPORT_TO=$(jq -r '.report_to' $CONFIG_FILE)
#BF16=$(jq -r '.bf16' $CONFIG_FILE)
LOGGING_FIRST_STEP=$(jq -r '.logging_first_step' $CONFIG_FILE)
USE_PEFT=$(jq -r '.use_peft' $CONFIG_FILE)
#LOAD_IN_8BIT=$(jq -r '.load_in_8bit' $CONFIG_FILE)
LORA_R=$(jq -r '.lora_r' $CONFIG_FILE)
LORA_ALPHA=$(jq -r '.lora_alpha' $CONFIG_FILE)
LORA_TARGET_MODULES=$(jq -r '.lora_target_modules' $CONFIG_FILE)
EVAL_STRATEGY=$(jq -r '.eval_strategy' $CONFIG_FILE)
UNDESIRABLE_WEIGHT=$(echo $CONFIG | jq -r '.learning_rate[]')
echo $LEARNING_RATES

echo "Running script"
# Build the command
for lr in $LEARNING_RATES; do
    echo "Running script with learning rate: $lr"
    #for undes_weight in $UNDESIRABLE_WEIGHT; do
    python src_kto.py --model_name_or_path $model_name \
                      --output_path $output_path \
                      --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
                      --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
                      --num_train_epochs $NUM_TRAIN_EPOCHS \
                      --learning_rate $lr \
                      --lr_scheduler_type=$LR_SCHEDULER_TYPE \
                      --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                      --logging_steps $LOGGING_STEPS \
                      --output_dir=$OUTPUT_DIR \
                      --warmup_ratio $WARMUP_RATIO \
                      --report_to $REPORT_TO \
                      --bf16 \
                      --logging_first_step \
                      --use_peft \
                      --load_in_8bit \
                      --lora_r $LORA_R \
                      --lora_alpha $LORA_ALPHA \
                      --temperature $temperature \
                      --eval_strategy $EVAL_STRATEGY \
                      --num_fold $num_fold
                      #--undesirable_weight $undes_weight \
      #done
    done
#--lora_target_modules $LORA_TARGET_MODULES
                  
# python scr_baseline_kto.py --model_name $model_name \
#                            --input_path $input_path \
#                            --output_path $output_path \
#                            --num_fold $num_fold \
#                            --temperature $temperature
