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

import sys
import os
import json
from dataclasses import dataclass, field
from typing import Optional, List

from collections import Counter


import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback
)
from trl import (
    SFTConfig, DPOConfig, KTOConfig, ModelConfig,
    get_peft_config, setup_chat_format,
    SFTTrainer, DPOTrainer, KTOTrainer
)
from peft import get_peft_model

# from intervention.reasoning import (
#     evaluate, evaluate_batch,
#     parse_output, 
# )

from utils.ut_run_llms import evaluate, evaluate_batch

from utils.ut_evaluation_utils import precision_recall_consistency
from utils.ut_processing_utils import prepare_prompt, parse_output, extract_final_answer
from utils.ut_evaluation_utils import extract_answer_compare_gt, evaluate_configuration_general

# ------------------
# Common argument definitions
# ------------------
@dataclass
class BaseArgs:
    model_name_or_path: str = field(
        default="",
        metadata={"help": "Base model name or path."}
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "Where to save models and outputs."}
    )
    trainer_type: str = field(
        default="KTO",
        metadata={"help": "Which trainer: 'KTO', 'SFT', or 'DPO'."}
    )


@dataclass
class ScriptArguments:
    output_path: str = field(metadata={"help": "Directory for outputs."})
    input_path: str = field(
        default="./data.json",
        metadata={"help": "Path to input JSON or JSON-lines file."}
    )
    num_fold: int = field(default=2, metadata={"help": "Number of folds."})
    temperature: float = field(default=0.3, metadata={"help": "Sampling temperature."})
    consistency_factor: int = field(default=5, metadata={"help": "Consistency repeats."})
    custom_map: bool = field(default=False, metadata={"help": "Use custom mapping."})
    valid_path: Optional[str] = None      # <-- new
    dataset_name: str = ""
# ------------------
# Evaluation utils & callback
# ------------------
# def custom_evaluate_batch(model, tokenizer, dataset, epoch, output_dir,
#                           temperature, fold, lr, seed, undes_weight):
#     results = []
#     batch_size = 2
#     max_new = 600
#     gen_args = dict(max_new_tokens=max_new, do_sample=True,
#                     num_beams=1, num_return_sequences=1,
#                     temperature=temperature, top_p=0.95)
#     for start in tqdm(range(0, len(dataset), batch_size)):
#         batch = dataset.iloc[start:start+batch_size]
#         # prompts = [
#         #     prepare_prompt(p, tokenizer, None, None)
#         #     if "<|eot_id|>" not in p else p
#         #     for p in batch.prompt.tolist()
#         # ]

#         prompts = batch.prompt.values.tolist() #row[1].prompt

#         prompts_processed = [
#             prepare_prompt(prompt, tokenizer, args.dataset_name, args.prompt_type) 
#             for prompt in batch.prompt.values.tolist()
#         ]
#         outs = evaluate_batch(prompts, model, tokenizer, **gen_args)
#         parsed = [parse_output(o, prompts[i], tokenizer)
#                   for i, o in enumerate(outs[0])]
#         for i, pout in enumerate(parsed):
#             row = batch.iloc[i]
#             #fa, pred = extract_final_answer(pout, cot=True, internal_cot=False)
#             results.append({
#                 'id': row.get('req_id', None),
#                 'prompt': prompts[i],
#                 'output': pout,
#                 'epoch': epoch
#             })
    
#     results = pd.DataFrame(results)

#     results = extract_answer_compare_gt(args, results)

#     metrics = {"accuracy": results.correct.value_counts().to_dict().get(True, 0) / curr_fold_results.shape[0]}
            
    
#     precision, recall = precision_recall_consistency(df_res)
#     metrics = dict(epoch=epoch, precision=precision,
#                    recall=recall, fold=fold)
#     os.makedirs(output_dir, exist_ok=True)
#     df_res.to_json(os.path.join(
#         output_dir, f"results_epoch_{epoch}_fold_{fold}.json"), orient='records', indent=4)
#     with open(os.path.join(output_dir, "metrics.jsonl"), 'a') as f:
#         f.write(json.dumps(metrics) + '\n')
#     return metrics

def custom_evaluate_batch(
        model,
        tokenizer,
        dataset,
        state,          # <-- pass Trainer's state so we get the epoch
        args,            # <-- ONE object instead of six scalars
        split_name
    ):
    epoch        = state.epoch
    output_dir   = args.output_dir
    temperature  = args.temperature
    #fold         = args.fold
    #lr           = args.learning_rate
    seed         = args.seed
    #undes_weight = args.undes_weight
    ...

    results = []
    batch_size = 2
    max_new = 600
    gen_args = dict(max_new_tokens=max_new, do_sample=True,
                    num_beams=1, num_return_sequences=1,
                    temperature=temperature, top_p=0.95)
    for start in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.iloc[start:start+batch_size]
    
        prompts = batch.prompt.values.tolist() #row[1].prompt

        prompts_processed = [
            prepare_prompt(prompt, tokenizer, args.dataset_name) 
            for prompt in batch.prompt.values.tolist()
        ]

        outs = evaluate_batch(prompts_processed, model, tokenizer, **gen_args)
        parsed = [parse_output(o, prompts_processed[i], tokenizer)
                  for i, o in enumerate(outs[0])]
        for i, pout in enumerate(parsed):
            row = batch.iloc[i]
            #fa, pred = extract_final_answer(pout, cot=True, internal_cot=False)
            results.append({
                'id': row.get('data_id', None),
                'prompt': prompts[i],
                'processed_prompt': prompts_processed[i],
                'output': pout,
                'epoch': epoch,
                'gt': row['gt']
            })
    args.prompt_type = "ab_cot"  # <-- reset prompt type after each batch
    results = extract_answer_compare_gt(args, results)

    results = pd.DataFrame(results)

    if args.dataset_name == "requirements_data":
        
        # Calculate precision and recall
        precision, recall = get_precision_recall(curr_fold_results)
        precision_consist, recall_consist = precision_recall_consistency(curr_fold_results)

        print("Precision: ", precision," Recall: ", recall)
        print("Precision consistency: ", precision_consist," Recall consistency: ", recall_consist)

        # Store the results
        precision_scores.append({"precision": precision, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas, "precision_consistency": precision_consist })
        recall_scores.append({"recall": recall, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas, "recall_consistency": recall_consist})

        metrics = {"precision": precision, "recall": recall, "precision_consistency": precision_consist, "recall_consistency": recall_consist}

        #save_results(args, heads , alphas, fold_index, results, metrics)

    else:

        fold_index = 0
        accuracy = results.correct.value_counts().to_dict().get(True, 0) / results.shape[0]
        metrics = {
            "epoch": epoch,
            "accuracy": accuracy,
            "fold": fold_index,
            "lerning_rate": args.learning_rate,
            "split": split_name
        }
        #save_results(args, heads , alphas, fold_index,  curr_fold_results, metrics)

        results.to_json(os.path.join(
        output_dir, f"results_{split_name}_epoch_{epoch}_fold_{fold_index}_lr_{args.learning_rate}.json"), orient='records', indent=4)
        with open(os.path.join(output_dir, "metrics.jsonl"), 'a') as f:
            f.write(json.dumps(metrics) + '\n')

        return metrics

# class CustomEvalCallback(TrainerCallback):
#     def __init__(self, eval_df, tokenizer, out_dir,
#                  fold, temperature, learning_rate,
#                  seed, undes_weight):
#         self.df = eval_df
#         self.tokenizer = tokenizer
#         self.out_dir = out_dir
#         self.fold = fold
#         self.temperature = temperature
#         self.lr = learning_rate
#         self.seed = seed
#         self.undes_weight = undes_weight
#     def on_epoch_end(self, args, state, control, model=None, **kwargs):
#         custom_evaluate_batch(
#             model, self.tokenizer, self.df,
#             state.epoch, self.out_dir,
#             self.temperature, self.fold,
#             self.lr, self.seed, self.undes_weight
#         )

class CustomEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_df,                     # ← always required
        tokenizer,
        train_df: Optional = None    # ← optional, default = None
    ):
        self.eval_df  = eval_df
        self.train_df = train_df     # may stay None
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # --- run on the train set only if we got one -----------------
        if self.train_df is not None:
            custom_evaluate_batch(
                model=model,
                tokenizer=self.tokenizer,
                dataset=self.train_df,
                state=state,
                args=args,
                split_name="train"
            )

        # --- always run on the validation set ------------------------
        custom_evaluate_batch(
            model=model,
            tokenizer=self.tokenizer,
            dataset=self.eval_df,
            state=state,
            args=args,
            split_name="eval"
        )

# ------------------
# Main
# ------------------
def main():
    parser = HfArgumentParser(BaseArgs)
    base_args, rem = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    ttype = base_args.trainer_type.upper()
    
    verbose = True 

    # load and split function
    def make_folds(df, num_fold):
        """
        Group samples by 'req_id' if present, otherwise each row is its own group.

        Returns a list of arrays, each containing the row indices for one fold.
        Special case num_fold==1: returns [test_rows, train_rows] with 20% test, 80% train.
        
        ## STRATIFIED NOT IMPLEMENTED AT THE MOMENT
        
        """
        # Determine group index for each row
        if 'req_id' in df.columns:
            reqs = df['req_id'].unique()
            idx_map = {rid: i for i, rid in enumerate(reqs)}
            inds = df['req_id'].map(idx_map).to_numpy()

        # Determine group index for each row
        elif 'data_id' in df.columns:
            reqs = df['data_id'].unique()
            idx_map = {rid: i for i, rid in enumerate(reqs)}
            inds = df['data_id'].map(idx_map).to_numpy()
        else:
            # No grouping column; treat each example separately
            inds = np.arange(len(df))

        unique_idxs = np.unique(inds)
        np.random.seed(0)
        np.random.shuffle(unique_idxs)

        # Special case: single fold as train/test split
        if num_fold == 1:
            split = int(len(unique_idxs) * 0.2)
            test_groups = unique_idxs[:split]
            train_groups = unique_idxs[split:]
            test_rows = np.where(np.isin(inds, test_groups))[0]
            train_rows = np.where(np.isin(inds, train_groups))[0]
            return [test_rows, train_rows]

        # General k-fold splitting
        fold_size = len(unique_idxs) // num_fold
        group_folds = [unique_idxs[i*fold_size:(i+1)*fold_size] for i in range(num_fold)]
        row_folds = []
        for gf in group_folds:
            rows = np.where(np.isin(inds, gf))[0]
            row_folds.append(rows)

        return row_folds

    print(ttype)
    # iterate trainer types
    for TrainerType in ['KTO', 'SFT', 'DPO']:
        if ttype != TrainerType:
            continue
        config_cls = {'KTO': KTOConfig, 'SFT': SFTConfig, 'DPO': DPOConfig}[TrainerType]
        TrainerCls = {'KTO': KTOTrainer, 'SFT': SFTTrainer, 'DPO': DPOTrainer}[TrainerType]
        parser = HfArgumentParser((ScriptArguments, config_cls, ModelConfig))
        script_args, train_args, model_args, leftovers = parser.parse_args_into_dataclasses(args=rem,
                                            return_remaining_strings=True)
        
        print(leftovers)
        
        train_args.output_dir = script_args.output_path
        model_args.model_name_or_path = base_args.model_name_or_path
        print(model_args.model_name_or_path)
        
        df = pd.read_json(script_args.input_path)

        if script_args.valid_path:          # a separate validation file was given
            test_df = pd.read_json(script_args.valid_path).reset_index(drop=True)
            train_df = df     
        else:
            # prepare folds
            folds = make_folds(df, script_args.num_fold)#, TrainerType)
            #print(folds)
            # loop folds
            for fold_idx, fold_rows in enumerate(folds):
                # determine train/test indices
                test_idx = fold_rows
                train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
                train_df = df.iloc[train_idx].reset_index(drop=True)
                print(train_df.iloc[0:1])
                
                test_df = df.iloc[test_idx].reset_index(drop=True)
                print(test_df.iloc[0:1])

        # apply consistency repeats only for KTO/DPO
        if TrainerType in ['KTO', 'DPO']:
            test_df = test_df.loc[test_df.index.repeat(script_args.consistency_factor)].reset_index(drop=True)
            
        # build HF datasets
        if TrainerType == 'SFT':
            train_ds = Dataset.from_dict({
                'prompt': train_df.prompt.tolist(),
                'completion': train_df.output.tolist()
            })
            # eval_ds = Dataset.from_dict({
            #     'prompt': test_df.prompt.tolist(),
            #     'completion': test_df.output.tolist()
            # })
        else:
            train_ds = Dataset.from_dict({
                'prompt': train_df.prompt.tolist(),
                'completion': train_df.output.tolist(),
                'label': train_df.correct.tolist()
            })
            #print(train_ds[0])
            eval_ds = Dataset.from_pandas(test_df)
            
            def concatenate_prompt_output(example):
                full_prompt = example['complete_inputs']#example['prompt'] + example['output']
                completion = example['output']
                label = example['correct']
                return {'text': full_prompt, 'completion': completion, 'label': label}
            
            eval_ds = eval_ds.map(concatenate_prompt_output)

        # load model/tokenizer
        
        if verbose: 
            print("Training Example")
            print(train_ds[0])

            print("Evaluation Example")
            #print(eval_ds[0])

            if TrainerType in ['KTO', 'DPO']:

                # Count values for the key 'label'
                label_counts = Counter(d["label"] for d in train_ds)
                print(label_counts)
                
                # Count values for the key 'label'
                label_counts = Counter(d["label"] for d in eval_ds)
                print(label_counts)

        if script_args.custom_map:
            print("Applying custom mapping")
            custom_device_map = {}
            for i in range(32):  # Assuming your model has 80 layers, adjust as needed
                if i < 17:
                    custom_device_map[f"model.layers.{i}"] = 0  # Assign to GPU 0
                else:
                    custom_device_map[f"model.layers.{i}"] = 1  # Assign to GPU 1

            # Include any other specific mappings if needed (e.g., for output layers)
            custom_device_map["model.embed_tokens.weight"]=0
            custom_device_map["model.norm.weight"] = 1       # Assign final normalization to GPU 1
            custom_device_map["lm_head"] = 1    # Assign output head to GPU 1

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=custom_device_map,
                use_cache=not train_args.gradient_checkpointing
            )

        else:

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                use_cache=not train_args.gradient_checkpointing
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            padding_side="left")
        
        if tokenizer.pad_token is None:
            #tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token = "<unk>"
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        # apply PEFT/LoRA only if requested
        if getattr(train_args, 'use_peft', False):
            peft_cfg = get_peft_config(model_args)
            model = get_peft_model(model, peft_cfg)

        # # set up evaluation callback
        # callback = CustomEvalCallback(
        #     test_df, tokenizer, train_args.output_dir,
        #     fold_idx, script_args.temperature,
        #     train_args.learning_rate, train_args.seed,
        #     getattr(train_args, 'undesirable_weight', None)
        # )

        # set up evaluation callback
        callback = CustomEvalCallback(
            eval_df=test_df,#.iloc[:2],
            tokenizer=tokenizer,          # callback keeps its own copy
            train_df=train_df.iloc[0:10]
        )
        
        train_args.temperature = script_args.temperature
        train_args.dataset_name = script_args.dataset_name
        # init trainer
        trainer = TrainerCls(
            model=model,
            args=train_args,
            train_dataset=train_ds,#.select(range(2)),
            #eval_dataset=eval_ds,
            #tokenizer=tokenizer,
            processing_class=tokenizer,  # This causes the error
            callbacks=[callback]
        )
        # train & save
        trainer.train()
        # trainer.save_model(train_args.output_dir)
        # if TrainerType == 'KTO':
        #     trainer.push_to_hub()
        return

    raise ValueError(f"Unknown trainer type: {base_args.trainer_type}")

if __name__ == "__main__":
    main()
