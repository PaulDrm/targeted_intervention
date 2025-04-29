import torch
import numpy as np
from einops import rearrange
import pickle
import os
from tqdm import tqdm
import pandas as pd
import ast 
import argparse
#from datasets import load_dataset
import json
import sys
sys.path.append('../')
#from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, load_model, prepare_prompt, run_llama_intervention, run_llama_intervention_batch, run_llama

sys.path.append('../app/')
from intervention.reasoning import eval_intervention, parse_output, extract_final_answer, evaluate 
import re

from ut_intervention_utils import run_llama

import llama

class ParseListOfLists(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            result = ast.literal_eval(values)
            if not all(isinstance(i, list) and len(i) == 2 for i in result):
                raise ValueError("Each sublist must contain exactly two elements.")
            setattr(namespace, self.dest, result)
        except ValueError as ve:
            raise argparse.ArgumentTypeError(f"Input error: {ve}")
        except:
            raise argparse.ArgumentTypeError("Input should be a list of lists with two strings each (e.g., [['11', '0'], ['13', '0']])")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def load_model_peft(model_name,peft_model_path ="", device="cuda:0"):

    print("Loading right tokenizer!")

    LOAD_8BIT = False #True
    BASE_MODEL = model_name

    #tokenizer = llama.LlamaTokenizer.from_pretrained(BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    #peft_model_path = "../sft_results/requirements_data/llama3_false_positives_1609_KTO_optimised_model/checkpoint-25"  # This should be the path to your PEFT model
    
    config = PeftConfig.from_pretrained(peft_model_path)

    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.cuda()
    if "openchat" in model.config._name_or_path:

        model.generation_config.pad_token_id = 0

    elif "Meta-Llama-3-8B-Instruct" in model.config._name_or_path:
        model.generation_config.pad_token_id = 128009#32007
        
        tokenizer.eos_token_id = 128009
        tokenizer.pad_token = tokenizer.eos_token
    
    elif "Llama-2-7b-chat-hf" in model.config._name_or_path:

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model 


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', help='model name')
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.5)
    #parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_center_of_mass', type=lambda x: (str(x).lower() == 'true'), help='Whether to use the center of mass or not')
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    #parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--output_path', type=str, help='output path')
    parser.add_argument('--dataset_name', type=str, default='requirements_data')
    parser.add_argument('--test_set_input_path', type=str)
    parser.add_argument('--prompt_type', type=str, default="open_ended")
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--consistency_factor', type=int, default=6)
    parser.add_argument('--peft_model_input_path', type=str)
    #parser.add_argument('--normalize_with_activations', type=lambda x: (str(x).lower() == 'true'), default='true')

    args = parser.parse_args()

    if not os.path.exists(f"{args.output_path}"):
        os.mkdir(f"{args.output_path}")

    # Print all arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    if args.dataset_name != "requirements_data":
        id_column = "data_id"
    else: 
        id_column = "req_id"
        #correct = [0 if value == "yes" else 1 for value in df.predict.values]
        #df.correct = correct

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(args.peft_model_input_path)

    tokenizer, model = load_model_peft(args.model_name, args.peft_model_input_path, device = "cuda:1")
    
    test_set = pd.read_json(args.test_set_input_path)
    print(test_set.shape)

    test_set.reset_index(drop=True, inplace=True)
    indexes = [test_set[test_set['req_id'] == req_id].index[0] for req_id in test_set.req_id.unique()]
    
    print(indexes)
    print(test_set.req_id.unique())
    # Repeat the list 10 times
    repeated_indexes = indexes * args.consistency_factor
    print(repeated_indexes)
    #test_set = train_set.loc[train_set.index.repeat(n)]
    test_set = test_set.loc[repeated_indexes]
    print(test_set.shape)

    results = run_llama(args, tokenizer, model, test_set)

    curr_fold_results = pd.DataFrame(results)

    curr_fold_results.to_json(f"{args.output_path}/ft_kto_results.json", orient='records', indent=4)
        
    #print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}")
    if args.prompt_type != "open_ended":
        print(curr_fold_results.predict.value_counts())

        with open(f'{args.output_path}/overall_results.txt', 'a') as f:
            # Redirect the print output to the file
            # print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}", file=f)
            print(f"For dataset fold {i} and alpha {args.alpha} and heads {head_string}", file=f)
            print(curr_fold_results.predict.value_counts(),file = f)
            print(curr_fold_results.final_answer.value_counts(),file = f)

        results = []
        counter = 0 
        
        value_counts = curr_fold_results.predict.value_counts().to_dict()
    
        # Path to your JSON file
        json_file_path = f'{args.output_path}/overall_results.json'
        
        # Load existing data from the JSON file
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []

        # Add the new data to the existing list (or create a new list if the file was not found)
        data.append(value_counts)

        # Write the updated data back to the JSON file
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

if __name__ == "__main__":
    main()   
    