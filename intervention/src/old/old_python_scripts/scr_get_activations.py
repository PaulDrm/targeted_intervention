import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from src.old.utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q
import llama
import pickle
import argparse

import pandas as pd 

from transformers import AutoTokenizer, AutoModelForCausalLM
HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'openchat': 'openchat/openchat_3.5'
}

def load_model(model_name): 

    LOAD_8BIT = False
    BASE_MODEL = model_name
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    if "openchat" in model.config._name_or_path:

        model.generation_config.pad_token_id = 0

    return tokenizer, model 

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir
    print(MODEL)
    # tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    # model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    # device = "cuda"
    tokenizer, model = load_model(MODEL)

    device  = "cuda"
    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, texts , labels, categories = formatter(dataset, tokenizer)
        with open(f'features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    all_layer_wise_activations = []
    all_head_wise_activations = []
    all_oproj_wise_activations = []
    print("Getting activations")
    for prompt in tqdm(prompts[0:100]):
        # layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        layer_wise_activations, head_wise_activations, o_proj_activations = get_llama_activations_bau(model, prompt, device)
        
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
        all_head_wise_activations.append(head_wise_activations[:,-1,:])
        all_oproj_wise_activations.append(o_proj_activations[:,-1,:])

    print(all_head_wise_activations[0].shape)

    pd.DataFrame({'prompts': texts, 'labels': labels, 'categories': categories}).to_json("truthful_gen_comparison.json", orient='records', indent =4)
    print("Saving labels")
    np.save(f'features/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'features/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'features/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()