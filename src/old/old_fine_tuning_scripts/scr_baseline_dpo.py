from trl import SFTTrainer, DPOTrainer
from transformers import TrainingArguments

#from unsloth import is_bfloat16_supported
import torch

import pandas as pd
import sys
sys.path.append('../')
from src.intervention.reasoning import load_model, eval_intervention, eval_intervention_batch, parse_output, extract_final_answer, evaluate 

#from honest_llama.ut_evaluation_utils import get_precision_recall

from tqdm import tqdm

import numpy as np

import os

import json

def custom_evaluate(model, tokenizer, eval_dataset, epoch, output_dir, temperature, fold):
    model.eval()
    MAX_NEW_TOKENS = 600
    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": temperature, # 0.8,
        "top_p": 0.95,
    }
    
    results = []
    for row in tqdm(eval_dataset.iterrows()):
        prompt = row[1]['prompt']
        output = evaluate(prompt, model, tokenizer, **generation_args)
        
        output = parse_output(output[0], prompt, tokenizer)
        
        
        #print(output[0])
        cot = True
        internal_cot = False
        final_answer, predict = extract_final_answer(output, cot=cot, internal_cot=internal_cot)
        gt = row[1]['final_answer'] if row[1]['correct'] else not row[1]['final_answer']
        results.append({
            'id': row[1][id_column],
            'req_id': row[1][id_column],
            "prompt": prompt,
            "output": output,
            "final_answer": final_answer,
            "predict": predict,
            "gt": gt,
            "epoch": epoch,
        })
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Compute metrics
    precision, recall = get_precision_recall(df_results)
    
    # Create metrics dictionary
    metrics = {
        "epoch": epoch,
        "precision": precision,
        "recall": recall,
        "fold": fold,
    }
    print(metrics)
    # Save results and metrics
    save_results(df_results, metrics, output_dir, epoch, fold)
    
    return metrics

def save_results(df_results, metrics, output_dir, epoch, fold):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results DataFrame
    results_file = os.path.join(output_dir, f"results_epoch_{epoch}_{fold}.json ")
    df_results.to_json(results_file, orient='records', indent=4)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    # Also save current epoch metrics separately for easy access
    current_metrics_file = os.path.join(output_dir, f"metrics_epoch_{epoch}_{fold}.json")
    with open(current_metrics_file, 'w') as f:
        json.dump(metrics, f)

from transformers import TrainerCallback

class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, output_dir, fold, eval_steps=1, temperature=0.8, ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.temperature = temperature
        self.fold = fold
    #def on_train_begin(self, args, state, control, **kwargs):
        #self.trainer = kwargs.get('trainer', None)

    def on_epoch_end(self, args, state, control, model, **kwargs):
        #logger.info(f"Epoch {state.epoch} has ended")
        metrics = custom_evaluate(model, self.tokenizer, self.eval_dataset, 
                                   state.epoch, self.output_dir, self.temperature, self.fold)
        

    # def on_step_end(self, args, state, control, model, **kwargs):
    #     # Evaluate the model every eval_steps steps
    #     if state.global_step % self.eval_steps == 0:
    #     #    control.should_evaluate = True
    #         metrics = custom_evaluate(model, self.tokenizer, self.eval_dataset, 
    #                                state.epoch, self.output_dir)
        
    # def on_evaluate(self, args, state, control, **kwargs):

    #     metrics = custom_evaluate(self.model, self.tokenizer, self.eval_dataset, 
    #                                state.epoch, self.output_dir)
        
    # def on_epoch_end(self, args, state, control, **kwargs):
        
    #     metrics = custom_evaluate(self.model, self.tokenizer, self.eval_dataset, 
    #                                state.epoch, self.output_dir)
    #     #self.trainer.log(metrics)
    #     if 'precision' in metrics:  # Or whichever metric you're using for best model
    #         self.trainer.state.best_metric = max(self.trainer.state.best_metric, metrics['precision'])

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama_7B', help='model name')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--input_path', type=str, help='input path')
parser.add_argument('--output_path', type=str, help='output path')
parser.add_argument('--dataset_name', type=str, help='dataset name', default='requirements_data')
parser.add_argument('--num_fold', type=int, default=1, help='num fold')
parser.add_argument('--temperature', type=float, default=0.8, help='temperature')
args = parser.parse_args()

if not os.path.exists(f"{args.output_path}"):
    os.mkdir(f"{args.output_path}")

# Print all arguments
print("Parsed arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")

df = pd.read_json(args.input_path)

if args.dataset_name != "requirements_data": 
    id_column = "data_id"

else: 
    id_column = "req_id"

column = "attentions" 

# set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
index_dic = {}
separated_activations = []
separated_labels = []
reqs_order = []

for req_id in df[id_column].unique():

    req_df = df[df[id_column] == req_id].index

    #req_ids.append(req_df)
    index_dic[req_id] = list(req_df)
    
    temp_activations = df[df[id_column] == req_id][column]
    activations = np.array([list(sample.values()) for sample in temp_activations.values])#.shape
    batch_length = len(temp_activations)
    dim = 128
    activations = np.reshape(activations, (batch_length, 32, 32, dim))

    temp_labels =[1 if label==True else 0 for label in df[df[id_column] == req_id]['correct'].values]
    separated_labels.append(temp_labels)
    separated_activations.append(activations)
    reqs_order.append(req_id)

## Set up random number generator
rng = np.random.default_rng(args.seed)

# Create indices for all samples
all_indices = np.arange(len(reqs_order))

# Shuffle indices
rng.shuffle(all_indices)

# Split indices into folds
fold_size = len(all_indices) // args.num_fold
fold_indices = [all_indices[i:i+fold_size] for i in range(0, len(all_indices), fold_size)]

print(f"Number of folds: {args.num_fold}")

tokenizer, base_model = load_model(args.model_name, load_in_8_bit=True)


# Iterate over all folds
for fold in range(args.num_fold):
    print(f"\nFold {fold + 1}/{args.num_fold}")
    if fold == 0:
        continue
    # Use current fold as validation set, and the rest as training set
    test_idxs = fold_indices[fold]
    train_idxs = np.concatenate([fold_indices[i] for i in range(args.num_fold) if i != fold])
    val_idxs = train_idxs
    print(f"Train set size: {len(train_idxs)}")
    print(f"Validation set size: {len(val_idxs)}")
    print(f"Test set size: {len(test_idxs)}")
    
    # Create datasets
    train_index_expanded = np.concatenate([list(index_dic.values())[i] for i in train_idxs])
    val_index_expanded = np.concatenate([list(index_dic.values())[i] for i in val_idxs])
    test_index_expanded = np.concatenate([list(index_dic.values())[i] for i in test_idxs])
    train_set_idxs = train_idxs
    val_set_idxs = test_idxs # val_idxs
    print(train_set_idxs)
    print(val_set_idxs)

    train_set = df.loc[train_index_expanded]
    print(train_set.shape)
    val_set = df.loc[val_index_expanded]
    test_set = df.loc[test_index_expanded]
    print(test_set.shape)
    #break

    from datasets import Dataset
    # Assuming you have a pandas DataFrame called 'df'
    train_dataset = train_set[train_set['correct'] == True]#[['prompt', 'output']]
    train_dataset = Dataset.from_pandas(train_dataset)

    # Define a function to concatenate prompt and output
    def concatenate_prompt_output(example):
        full_prompt = example['complete_inputs']#example['prompt'] + example['output']
        return {'text': full_prompt}

    # Apply the function to each example in the dataset
    train_dataset = train_dataset.map(concatenate_prompt_output)
    val_set = test_set#[['prompt', 'output']]
    eval_dataset = Dataset.from_pandas(val_set)
    eval_dataset = eval_dataset.map(concatenate_prompt_output)

    output_dir = args.output_path

    custom_evaluate(base_model, tokenizer, test_set, -1, output_dir,args.temperature, fold)
    # from unsloth import FastLanguageModel 
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = args.model_name,
    #     max_seq_length = max_seq_length,
    #     dtype = None,
    #     #load_in_8bit = True,
    # )
    # tokenizer.pad_token_id = 0
    # # Do model patching and add fast LoRA weights
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 16,
    #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                       "gate_proj", "up_proj", "down_proj",],
    #     lora_alpha = 16,
    #     lora_dropout = 0, # Supports any, but = 0 is optimized
    #     bias = "none",    # Supports any, but = "none" is optimized
    #     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    #     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    #     random_state = 3407,
    #     max_seq_length = max_seq_length,
    #     use_rslora = False,  # We support rank stabilized LoRA
    #     loftq_config = None, # And LoftQ
    # )

    from peft import prepare_model_for_kbit_training
    from peft import LoraConfig, get_peft_model

    #model = base_model
    

    base_model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(base_model)

    config = LoraConfig(
        r=16, 
        lora_alpha=16, 
        target_modules="all-linear", 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, config)

    custom_evaluate(model, tokenizer, test_set, fold, output_dir, args.temperature, 0)

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    print_trainable_parameters(model)

    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps = 12,
        warmup_steps = 5,
        max_steps=-1,  # Set to -1 to use num_train_epochs instead
        num_train_epochs=5,  # Train for 5 epochs
        #max_steps = 60,
        learning_rate = 2e-4,
        fp16 = False,# not is_bfloat16_supported(),
        bf16 = True,#is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        #eval_strategy="steps",
        #eval_steps=1,  # Evaluate every 500 steps
        eval_strategy="epoch",  # Evaluate after each epoch
        #save_strategy="epoch",  # Save model after each epoch
    )
    # eval_strategy="no",  # Disable built-in evaluation strategy
    # save_strategy="no",  # Disable built-in saving strategy
    # eval_strategy="steps",
    # eval_steps=1,  # Evaluate every 500 steps
    #eval_strategy="steps",
        #eval_strategy="no",  # Disable built-in evaluation strategy
        #eval_steps=1,  # Evaluate every 500 steps
        # load_best_model_at_end = True,
        # metric_for_best_model = "exact_match",  # Or whichever metric you prefer
    custom_callback = CustomEvalCallback(test_set, tokenizer, output_dir, fold, temperature=args.temperature)

    # # Now create the trainer
    # dpo_trainer = DPOTrainer(
    # model = model, 
    #     ref_model = base_model
    #     tokenizer = tokenizer,
    #     train_dataset = train_dataset,
    #     eval_dataset = eval_dataset,
    #     dataset_text_field = "text",
    #     max_seq_length = max_seq_length,
    #     dataset_num_proc = 2,
    #     packing = False,
    #     args = training_args,
    #     callbacks = [custom_callback],
    # )
    
    trainer.train()