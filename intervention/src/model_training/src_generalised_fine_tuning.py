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
Run the KTO training script with the commands below. In general, the optimal configuration for KTO 
will be similar to that of DPO.

# Full training example:
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

# QLoRA example:
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
import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional: if your project structure absolutely needs it
# sys.path.append('../')

from typing import Optional
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
    EarlyStoppingCallback
)

# KTO imports
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format

# PEFT imports
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

# Local utility imports
from intervention.reasoning import (evaluate, evaluate_batch)

from utils.ut_evaluation_utils import precision_recall_consistency, extract_final_answer
from utils.ut_processing_utils import prepare_prompt, parse_output

# =========== Utility functions for evaluation and saving results =========== #

def custom_evaluate(
    model,
    tokenizer,
    eval_dataset,
    epoch,
    output_dir,
    temperature,
    fold,
    lr,
    seed,
    undes_weight,
    evaluation_fn,
    id_column="data_id",
):
    """
    Performs evaluation by iterating through `eval_dataset` one row at a time.
    """
    model.eval()
    MAX_NEW_TOKENS = 600
    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": temperature,  # for sampling
        "top_p": 0.95,
    }
    
    results = []
    for _, row in tqdm(eval_dataset.iterrows()):
        prompt = row["prompt"]
        output = evaluate(prompt, model, tokenizer, **generation_args)
        parsed_output = parse_output(output[0], prompt, tokenizer)

        # Extract final answer
        final_answer, predict = extract_final_answer(
            parsed_output,
            cot=True,
            internal_cot=False
        )
        # Ground truth
        gt = row["final_answer"] if row["correct"] else not row["final_answer"]

        results.append({
            "id": row[id_column],
            "data_id": row[id_column],
            "prompt": prompt,
            "output": parsed_output,
            "final_answer": final_answer,
            "predict": predict,
            "gt": gt,
            "epoch": epoch,
        })
    
    df_results = pd.DataFrame(results)

    metrics = evaluation_fn(df_results)

    metrics['epoch']
    metrics['fold']

    print(metrics)
    save_results(df_results, metrics, output_dir, epoch, fold, lr, seed, undes_weight)
    
    return metrics

def custom_evaluate_batch(
    model,
    tokenizer,
    dataset,
    epoch,
    output_dir,
    temperature,
    fold,
    lr,
    seed,
    undes_weight,
    evaluation_fn,
    id_column="data_id",
):
    """
    Performs evaluation in batches for efficiency.
    """
    model.eval()
    MAX_NEW_TOKENS = 600
    batch_size = 2  # Adjust as needed

    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": temperature,
        "top_p": 0.95,
    }

    results = []
    for start_idx in tqdm(range(0, len(dataset), batch_size)):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset.iloc[start_idx:end_idx]
        
        # Prepare prompts
        prompts = [
            prepare_prompt(p, tokenizer)
            if "<|eot_id|>" not in p else p
            for p in batch["prompt"].values.tolist()
        ]

        outputs = evaluate_batch(prompts, model, tokenizer, **generation_args)
        parsed_outputs = [
            parse_output(out, prompts[i], tokenizer)
            for i, out in enumerate(outputs[0])
        ]

        for i, parsed_output in enumerate(parsed_outputs):
            row = batch.iloc[i]
            final_answer, predict = extract_final_answer(
                parsed_output,
                cot=True,
                internal_cot=False
            )
            results.append({
                "id": row[id_column],
                "req_id": row[id_column],
                "prompt": prompts[i],
                "output": parsed_output,
                "final_answer": final_answer,
                "predict": predict,
                "epoch": epoch,
            })    
    
    df_results = pd.DataFrame(results)
    df_results = pd.DataFrame(results)

    metrics = evaluation_fn(df_results)

    metrics['epoch']
    metrics['fold']
            
    print(metrics)
    save_results(df_results, metrics, output_dir, epoch, fold, lr, seed, undes_weight)
    
    return metrics

def save_results(df_results, metrics, output_dir, epoch, fold, lr, seed, undes_weight):
    """
    Saves results and metrics to the specified `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Results file
    results_file = os.path.join(
        output_dir,
        f"results_epoch_{epoch}_fold_{fold}_lr_{lr}_seed_{seed}_weight_{undes_weight}.json"
    )
    df_results.to_json(results_file, orient="records", indent=4)
    
    # Metrics file (appended)
    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")
    
    # Current epoch metrics file
    current_metrics_file = os.path.join(
        output_dir,
        f"metrics_epoch_{epoch}_fold_{fold}_lr_{lr}_seed_{seed}_weight_{undes_weight}.json"
    )
    with open(current_metrics_file, "w") as f:
        json.dump(metrics, f)

# =========== Custom Trainer Callbacks =========== #
class CustomEvalCallback(TrainerCallback):
    """
    Runs evaluation at the end of each epoch.
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        output_dir,
        fold,
        eval_steps=1,
        temperature=0.8,
        learning_rate=None,
        seed=42,
        undes_weight=None,
        use_batch_eval=True
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.temperature = temperature
        self.fold = fold
        self.learning_rate = learning_rate
        self.seed = seed
        self.undes_weight = undes_weight
        self.use_batch_eval = use_batch_eval

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.use_batch_eval:
            custom_evaluate_batch(
                model=model,
                tokenizer=self.tokenizer,
                dataset=self.eval_dataset,
                epoch=state.epoch,
                output_dir=self.output_dir,
                temperature=self.temperature,
                fold=self.fold,
                lr=self.learning_rate,
                seed=self.seed,
                undes_weight=self.undes_weight
            )
        else:
            custom_evaluate(
                model=model,
                tokenizer=self.tokenizer,
                eval_dataset=self.eval_dataset,
                epoch=state.epoch,
                output_dir=self.output_dir,
                temperature=self.temperature,
                fold=self.fold,
                lr=self.learning_rate,
                seed=self.seed,
                undes_weight=self.undes_weight
            )

class StopCallback(TrainerCallback):
    """
    Stops training after a set number of epochs.
    """
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if state.epoch >= 2:  # Example condition
            control.should_training_stop = True

# =========== Argument classes =========== #

@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """
    output_path: str = field(metadata={"help": "The output directory"})
    num_fold: int = field(default=2, metadata={"help": "Number of folds for cross-validation"})
    temperature: float = field(default=0.3, metadata={"help": "Sampling temperature"})
    consistency_factor: int = field(default=5, metadata={"help": "Consistency factor for repeated evaluations"})

@dataclass
class OptionalArguments:
    """
    Additional arguments to make PEFT and custom device mapping optional.
    """
    use_peft: bool = field(default=False, metadata={"help": "Use LoRA/PEFT fine-tuning"})
    use_custom_device_map: bool = field(default=False, metadata={"help": "Use a custom device map instead of auto"})
    input_path: Optional[str] = field(
        default="../datasets/requirements_data/dataframe_llama3b_cot_moon_12072024_attentions_false_positives_corout.json",
        metadata={"help": "Default path to input JSON data"}
    )

# =========== Main script logic =========== #

def main():
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig, OptionalArguments))
    args, kto_args, model_args, opt_args = parser.parse_args_into_dataclasses()

    print(f"Basic script arguments: {args}")
    print(f"KTO arguments: {kto_args}")
    print(f"Model arguments: {model_args}")
    print(f"Optional arguments: {opt_args}")

    # Set the final output directory from ScriptArguments
    kto_args.output_dir = args.output_path
    input_path = opt_args.input_path

    # Read and prep data
    df = pd.read_json(input_path)
    id_column = "req_id"
    column = "attentions"

    # Seed setting for reproducibility
    torch.manual_seed(kto_args.seed)
    np.random.seed(kto_args.seed)
    torch.cuda.manual_seed_all(kto_args.seed)

    # Make sure output dir exists
    if not os.path.exists(kto_args.output_dir):
        os.makedirs(kto_args.output_dir)

    # Splitting data for cross-validation
    index_dic = {}
    separated_activations = []
    separated_labels = []
    reqs_order = []

    for req_id in df[id_column].unique():
        req_df = df[df[id_column] == req_id].index
        index_dic[req_id] = list(req_df)
        temp_activations = df[df[id_column] == req_id][column]
        # (Below lines show how you might handle attentions, if needed)
        activations = np.array([list(sample.values()) for sample in temp_activations.values])
        activations = np.reshape(activations, (len(temp_activations), 32, 32, 128))

        temp_labels = [
            1 if label is True else 0
            for label in df[df[id_column] == req_id]["correct"].values
        ]
        separated_labels.append(temp_labels)
        separated_activations.append(activations)
        reqs_order.append(req_id)

    rng = np.random.default_rng(kto_args.seed)
    all_indices = np.arange(len(reqs_order))

    # For demonstration, do a simple 2-fold split (could be expanded to your full cross-val)
    if args.num_fold == 1:
        train_size = int(len(all_indices) * 0.8)
        fold_indices = [all_indices[train_size:], all_indices[:train_size]]
    else:
        fold_size = len(all_indices) // args.num_fold
        fold_indices = [all_indices[i : i + fold_size] for i in range(0, len(all_indices), fold_size)]

    # Iterate over folds
    for fold in range(args.num_fold):
        print(f"\nFold {fold + 1}/{args.num_fold}")

        test_idxs = fold_indices[fold]
        train_idxs = np.concatenate(
            [fold_indices[i] for i in range(len(fold_indices)) if i != fold]
        )
        
        # For demonstration, use train_idxs again as validation set 
        # (Adjust to your actual logic if you have a separate dev set)
        val_idxs = train_idxs  

        train_index_expanded = np.concatenate([index_dic[reqs_order[i]] for i in train_idxs])
        val_index_expanded   = np.concatenate([index_dic[reqs_order[i]] for i in val_idxs])
        test_index_expanded  = np.concatenate([index_dic[reqs_order[i]] for i in test_idxs])

        train_set = df.loc[train_index_expanded]
        val_set   = df.loc[val_index_expanded]
        test_set  = df.loc[test_index_expanded]

        # Prepare dataset for KTO
        kto_dataset_dict = {
            "prompt": train_set["prompt"].tolist(),
            "completion": train_set["output"].tolist(),
            "label": train_set["correct"].tolist()
        }
        train_dataset = Dataset.from_dict(kto_dataset_dict)

        # Make a repeated test set for consistency factor
        test_set.reset_index(drop=True, inplace=True)
        indexes = [test_set[test_set[id_column] == rid].index[0] for rid in test_set[id_column].unique()]
        repeated_indexes = indexes * args.consistency_factor
        test_set = test_set.loc[repeated_indexes].copy()
        val_set  = test_set  # using test set as eval dataset here

        eval_dataset = Dataset.from_pandas(val_set)

        def concatenate_prompt_output(example):
            full_prompt = example["prompt"]
            completion  = example["completion"]
            label       = example["label"]
            return {"text": full_prompt, "completion": completion, "label": label}

        eval_dataset = eval_dataset.map(concatenate_prompt_output)

        # =========== Load Model and Tokenizer =========== #
        if opt_args.use_custom_device_map:
            # Example custom device map. Adjust layer indices as needed
            custom_device_map = {}
            for i in range(32):
                if i < 13:
                    custom_device_map[f"model.layers.{i}"] = 0
                else:
                    custom_device_map[f"model.layers.{i}"] = 1
            custom_device_map["model.embed_tokens.weight"] = 0
            custom_device_map["model.norm.weight"] = 1
            custom_device_map["lm_head"] = 1

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=custom_device_map,
                use_cache=not kto_args.gradient_checkpointing
            )
        else:
            # Default to device_map="auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto", 
                use_cache=not kto_args.gradient_checkpointing
            )
            
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        # (Optional) Evaluate baseline if you want
        # custom_evaluate(
        #     model, tokenizer, test_set, -1, kto_args.output_dir, args.temperature,
        #     fold, 0, kto_args.seed, 0
        # )

        # =========== Wrap model with PEFT if needed =========== #
        if opt_args.use_peft:
            
            # If you want a different set of default LoRA modules, adjust here
            model_args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            peft_config = get_peft_config(model_args)
            model = get_peft_model(model, peft_config)

        # =========== Trainer Setup =========== #
        kto_args.output_dir = args.output_path
        kto_args.evaluation_strategy = "epoch"

        custom_callback = CustomEvalCallback(
            eval_dataset=val_set,
            tokenizer=tokenizer,
            output_dir=kto_args.output_dir,
            fold=fold,
            temperature=args.temperature,
            learning_rate=kto_args.learning_rate,
            seed=kto_args.seed,
            undes_weight=kto_args.undesirable_weight,
            use_batch_eval=True
        )

        kto_trainer = KTOTrainer(
            model=model,
            args=kto_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[custom_callback]
        )

        kto_trainer.train()
        kto_trainer.save_model(kto_args.output_dir)
        kto_trainer.push_to_hub()

if __name__ == "__main__":
    main()
