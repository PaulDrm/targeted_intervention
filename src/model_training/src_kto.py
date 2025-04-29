# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run the KTO training script with the commands below. In general, the optimal configuration for KTO will be similar to that of DPO.

# Full training:
python examples/scripts/kto.py \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step

# QLoRA:
python kto.py \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=2 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 24 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model-lora \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_8bit \
    --lora_r=16 \
    --lora_alpha=16 \
    --lora_target_modules=all-linear 
    
"""
import sys
sys.path.append('../')
from dataclasses import dataclass

from datasets import load_dataset
from datasets import Dataset
    
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers import EarlyStoppingCallback
from transformers import TrainerCallback

from intervention.reasoning import load_model, eval_intervention, eval_intervention_batch, parse_output, extract_final_answer, evaluate, evaluate_batch

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format

from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

import torch

import torch.optim as optim

import numpy as np

import pandas as pd 

from tqdm import tqdm 

from utils.ut_evaluation_utils import precision_recall_consistency
import os 
import json

from utils.ut_processing_utils import prepare_prompt

from dataclasses import dataclass, field
from typing import Optional


def custom_evaluate(model, tokenizer, eval_dataset, epoch, output_dir, temperature, fold, lr, seed, undes_weight):
    
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
    precision, recall = precision_recall_consistency(df_results)
    
    # Create metrics dictionary
    metrics = {
        "epoch": epoch,
        "precision": precision,
        "recall": recall,
        "fold": fold,
    }

    print(metrics)
    # Save results and metrics
    save_results(df_results, metrics, output_dir, epoch, fold, lr, seed, undes_weight)
    
    return metrics


def custom_evaluate_batch(model, tokenizer, dataset, epoch, output_dir, temperature, fold, lr, seed, undes_weight):

    num_heads = 32

    batch_size = 2

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

    for start_idx in tqdm(range(0, len(dataset), batch_size)):

        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset.iloc[start_idx:end_idx]
        
        prompts = batch.prompt.values.tolist() #row[1].prompt

        prompts = [
            prepare_prompt(prompt, tokenizer, args.dataset_name, args.prompt_type) 
            if "<|eot_id|>" not in prompt 
            else prompt 
            for prompt in batch.prompt.values.tolist()
        ]

        output = evaluate_batch(prompts, model, tokenizer, **generation_args)

        # Using list comprehension to process the outputs
        new_outputs = [parse_output(out, prompts[i], tokenizer) for i, out in enumerate(output[0])]
        

        for i, new_output in enumerate(new_outputs):

            row = batch.iloc[i]
            prompt = prompts[i]
            
            #print(output[0])
            cot = True
            internal_cot = False

            final_answer, predict = extract_final_answer(new_output, cot=cot, internal_cot=internal_cot)
        
            results.append({
                'id': row[id_column],
                'req_id': row[id_column],
                "prompt": prompt,
                "output": new_output,
                "final_answer": final_answer,
                "predict": predict,
                "epoch": epoch,
            })    
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Compute metrics
    precision, recall = precision_recall_consistency(df_results)
    
    # Create metrics dictionary
    metrics = {
        "epoch": epoch,
        "precision": precision,
        "recall": recall,
        "fold": fold,
    }

    print(metrics)
    # Save results and metrics
    save_results(df_results, metrics, output_dir, epoch, fold, lr, seed, undes_weight)
    
    return metrics


def save_results(df_results, metrics, output_dir, epoch, fold, lr, seed, undes_weight):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results DataFrame
    results_file = os.path.join(output_dir, f"results_epoch_{epoch}_fold_{fold}_lr_{lr}_seed_{seed}_weight_{undes_weight}.json ")
    df_results.to_json(results_file, orient='records', indent=4)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    # Also save current epoch metrics separately for easy access
    current_metrics_file = os.path.join(output_dir, f"metrics_epoch_{epoch}_fold_{fold}_lr_{lr}_seed_{seed}_weight_{undes_weight}.json")
    with open(current_metrics_file, 'w') as f:
        json.dump(metrics, f)

class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, output_dir, fold, eval_steps=1, temperature=0.8, learning_rate=None, seed=42, undes_weight=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.temperature = temperature
        self.fold = fold
        self.learning_rate = learning_rate
        self.seed = seed
        self.undes_weight = undes_weight
    #def on_train_begin(self, args, state, control, **kwargs):
        #self.trainer = kwargs.get('trainer', None)

    def on_epoch_end(self, args, state, control, model, **kwargs):
        #logger.info(f"Epoch {state.epoch} has ended")
        #metrics = custom_evaluate(model, self.tokenizer, self.eval_dataset, 
        #                           state.epoch, self.output_dir, self.temperature, self.fold, self.learning_rate, self.seed, self.undes_weight)
        metrics = custom_evaluate_batch(model, self.tokenizer, self.eval_dataset, 
                                   state.epoch, self.output_dir, self.temperature, self.fold, self.learning_rate, self.seed, self.undes_weight)

class StopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if state.epoch >= 2:  # Stop after 2 epochs
            control.should_training_stop = True

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    #dataset_name: str = "trl-lib/kto-mix-14k"
    #input_path: str = ""
    output_path: str = field(metadata={"help": "The output directory"})
    #overwrite_output_dir: bool = field(default=False, metadata={"help": "Overwrite the content of the output directory"})
    num_fold: int = field(default=2, metadata={"help": "Number of folds for cross-validation"})
    temperature: float = field(default=0.3, metadata={"help": "Sampling temperature"})
    #seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    consistency_factor: int = field(default=5, metadata={"help": "Consistency factor for intervention evaluation"})

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    args, kto_args, model_args = parser.parse_args_into_dataclasses()
    model_args.lora_target_modules =["q_proj", "k_proj", "v_proj", "o_proj"] #'all-linear' #["q_proj", "k_proj", "v_proj", "o_proj"]
                     # "gate_proj", "up_proj", "down_proj"]
    print(f"Arguments: {args}")
    print(f"Model arguments: {model_args}")
    print(f"KTO arguments: {kto_args}")
    
    print(f"Model output_dir: {kto_args.output_dir}")
    kto_args.output_dir = args.output_path
    args.input_path = "../datasets/requirements_data/dataframe_llama3b_cot_moon_12072024_attentions_false_positives_corout.json"
    #args.num_fold = 1 
    #args.temperature = 0.3
    df = pd.read_json(args.input_path)
    id_column = "req_id"
    column = "attentions" 
    #kto_args.seed =42
    # set seeds
    torch.manual_seed(kto_args.seed)
    np.random.seed(kto_args.seed)
    torch.cuda.manual_seed_all(kto_args.seed)

    #args.consistency_factor = 5
    
    # check if output directory exists
    if not os.path.exists(kto_args.output_dir):
        print("Model evaluation already exists. Skipping training.")
        os.makedirs(kto_args.output_dir)

    

    baseline =False
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

    # Set up random number generator
    rng = np.random.default_rng(kto_args.seed)

    # Create indices for all samples
    all_indices = np.arange(len(reqs_order))

    # Shuffle indices
    #rng.shuffle(all_indices)

    # Split indices into folds
    if args.num_fold == 1:
        train_size = int(len(all_indices) * 0.8)
        fold_indices = [
            all_indices[train_size:], # 40% for validation/test
            all_indices[:train_size]  # 60% for training   
        ]

        gts = pd.read_json("../datasets/requirements_data/requirements_gt_1510.json")

        gt_labels = gts[gts['req_id'].isin(reqs_order)]['gt'].values

        num_folds = 2 #args.num_fold# Number of folds

        # Organize indices by class
        class_indices = {}
        for idx, label in zip(all_indices, gt_labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Shuffle indices within each class
        for indices in class_indices.values():
            rng.shuffle(indices)

        # Split indices into stratified folds
        stratified_folds = [[] for _ in range(num_folds)]
        for indices in class_indices.values():
            fold_size = len(indices) // num_folds
            remainder = len(indices) % num_folds

            start = 0
            for i in range(num_folds):
                end = start + fold_size #+ (1 if i < remainder else 0)
                stratified_folds[i].extend(indices[start:end])
                start = end

        fold_indices = stratified_folds
        #print(stratified_folds[1])
        print(gts[gts['req_id'].isin([reqs_order[i] for i in stratified_folds[0]])]['gt'].value_counts())
        print(gts[gts['req_id'].isin([reqs_order[i] for i in stratified_folds[1]])]['gt'].value_counts())

    else:

        fold_size = len(all_indices) // args.num_fold
        fold_indices = [all_indices[i:i+fold_size] for i in range(0, len(all_indices), fold_size)]

    # formatted_dataset = dataset.map(format_dataset)
    # Assuming you have a pandas DataFrame called 'df'

    # Iterate over all folds
    for fold in range(args.num_fold):

        print(f"\nFold {fold + 1}/{args.num_fold}")
        #if fold == 0:
        #    continue

        # Use current fold as validation set, and the rest as training set
        test_idxs = fold_indices[fold]
        #train_idxs = np.concatenate([fold_indices[i] for i in range(args.num_fold) if i != fold])
        train_idxs = np.concatenate([fold_indices[i] for i in range(len(fold_indices)) if i != fold])
        
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


    kto_dataset_dict = {
        'prompt': train_set['prompt'].tolist(),
        'completion': train_set['output'].tolist(),
        'label': train_set['correct'].tolist()}

    # Load a pretrained model
    print(f"Load first model...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path, torch_dtype=torch.bfloat16, device_map = 'cpu')#, load_in_8bit=LOAD_8BIT)#, trust_remote_code=model_args.trust_remote_code #
    # model.to("cuda:1")

    custom_device_map = {}

    for i in range(32):  # Assuming your model has 80 layers, adjust as needed
        if i < 13:
            custom_device_map[f"model.layers.{i}"] = 0  # Assign to GPU 0
        else:
            custom_device_map[f"model.layers.{i}"] = 1  # Assign to GPU 1

    # Include any other specific mappings if needed (e.g., for output layers)
    custom_device_map["model.embed_tokens.weight"]=0
    custom_device_map["model.norm.weight"] = 1       # Assign final normalization to GPU 1
    custom_device_map["lm_head"] = 1    # Assign output head to GPU 1

    # Load model for full fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=custom_device_map, 
        use_cache=not kto_args.gradient_checkpointing
    )
    ## #device_map="auto",
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

        #tokenizer.eos_token_id = 128009
        
        model.generation_config.pad_token_id = tokenizer.pad_token_id#128009#32007
        

    if baseline:
        custom_evaluate(model, tokenizer, test_set, -1, kto_args.output_dir,args.temperature, fold, 0, kto_args.seed,0)

    peft_config = get_peft_config(model_args)
    #print(peft_config)
    # get peft model with the given config
    model = get_peft_model(model, peft_config)
    
    def print_peft_device_map(model):
        print("PeftModel Device Map:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.device}")
    
    #print_peft_device_map(model)
    
    #print(model.device)
    #print(isinstance(model, PeftModel))

    #print(f"Load second model...")
    #ref_model = AutoModelForCausalLM.from_pretrained(
    #    model_args.model_name_or_path, torch_dtype=torch.float16, load_in_4bit=LOAD_8BIT, device_map = 'auto')#, trust_remote_code=model_args.trust_remote_code
    #ref_model.to("cuda:1")
    #print(model.device, ref_model.device)
    
    train_dataset = Dataset.from_dict(kto_dataset_dict)
    #train_dataset = Dataset.from_pandas(train_dataset)

    #Define a function to concatenate prompt and output
    def concatenate_prompt_output(example):
        full_prompt = example['complete_inputs']#example['prompt'] + example['output']
        completion = example['output']
        label = example['correct']
        return {'text': full_prompt, 'completion': completion, 'label': label}

    # # Apply the function to each example in the dataset
    # train_dataset = train_dataset.map(concatenate_prompt_output)

    test_set.reset_index(drop=True, inplace=True)
    indexes = [test_set[test_set['req_id'] == req_id].index[0] for req_id in test_set.req_id.unique()]
    
    print(indexes)
    print(test_set.req_id.unique())
    # Repeat the list 10 times
    repeated_indexes = indexes * args.consistency_factor
    print(repeated_indexes)
    #test_set = train_set.loc[train_set.index.repeat(n)]
    test_set = test_set.loc[repeated_indexes]
    print("Test set shape is: ", test_set.shape)
    #print(test_set.shape)

    val_set = test_set#[['prompt', 'output']]

    eval_dataset = Dataset.from_pandas(val_set)
    eval_dataset = eval_dataset.map(concatenate_prompt_output)

    output_dir_results = args.output_path#"../sft_results/requirements_data/llama3_false_positives_0609_stratified_lr_sweep" #args.output_path #
    # If we are aligning a base model, we use ChatML as the default template
    #if tokenizer.chat_template is None:
    #    model, tokenizer = setup_chat_format(model, tokenizer)

    #Load the dataset
    #dataset = load_dataset(args.dataset_name)

    # Apply chat template
    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
        return example

    #formatted_dataset = dataset.map(format_dataset)
    #print(96*"=")
    #print(f"Train set example: {formatted_dataset['train'][0]}")
    #print(96*"=")
    print(train_dataset[0])

    #kto_args.per_device_train_batch_size = 1
    #kto_args.per_device_eval_batch_size = 4
    #kto_args.gradient_accumulation_steps = 16
    # kto_args.num_train_epochs=5  # Train for 5 epochs
    # kto_args.logging_steps = 1
    #max_steps = 60,
    #kto_args=learning_rate = 2e-4,
    #kto_args.desirable_weight=1.0
    #kto_args.undesirable_weight=2.4
    #kto_args.save_strategy = "epoch"

    
    #kto_args.optim = "rmsprop"
    #kto_args.fp16 = True
    #kto_args.beta=0.05
    #kto_args.load_best_model_at_end=True  # This is the key to avoid the error
    
    kto_args.evaluation_strategy = "epoch"
    #kto_args.eval_steps=4

    print("Undesirable weight:", kto_args.undesirable_weight)
    print("Max length:", kto_args.max_length)
    print(f"Max prompt length: {kto_args.max_prompt_length}")
    print("Optimizer:", kto_args.optim)

    custom_callback = CustomEvalCallback(test_set, tokenizer, output_dir_results, fold, 
                                         temperature=args.temperature, learning_rate=kto_args.learning_rate, seed=kto_args.seed, undes_weight=kto_args.undesirable_weight)
    
    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        #ref_model,
        args=kto_args,
        #train_dataset=formatted_dataset["train"],
        #eval_dataset=formatted_dataset["test"],
        
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #tokenizer=tokenizer,
        processing_class=tokenizer,  # This causes the error
        #peft_config=get_peft_config(model_args),
        callbacks = [custom_callback],        
    )

    # Train and push the model to the Hub
    kto_trainer.train()
    kto_trainer.save_model(kto_args.output_dir)
    kto_trainer.push_to_hub()

