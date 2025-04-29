import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset

from moon_utils import load_model, generate_majority_predictions

import sys
sys.path.append('../')
from src.old.utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions
import llama


def load_model(model_name):

    LOAD_8BIT = True
    BASE_MODEL = model_name

    tokenizer = llama.LlamaTokenizer.from_pretrained(BASE_MODEL)
    model = llama.LlamaForCausalLM.from_pretrained(BASE_MODEL, low_cpu_mem_usage=True, load_in_8bit=LOAD_8BIT, torch_dtype=torch.float16, device_map="auto")

    if "openchat" in model.config._name_or_path:

        model.generation_config.pad_token_id = 0

    return tokenizer, model 

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.5)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()

    df_openchat = pd.read_json("../app/dataframe_open_chat_cot_moon_18012024_attentions.json")
    predict_openchat = generate_majority_predictions(df_openchat)

    df_gpt4 = pd.read_json("../app/results_gpt4_cot_moon_complete.json")
    predict_gpt4 = generate_majority_predictions(df_gpt4)

    # Merge the DataFrames on the 'req_id' column
    merged_df = pd.merge(predict_gpt4, predict_openchat, on='req_id', suffixes=('_gpt4', '_openchat'))

    # Compare the values in the 'majority_predict' column
    merged_df['is_same'] = merged_df['majority_predict_gpt4'] == merged_df['majority_predict_openchat']
    merged_df.head(5)

    df1 = df_openchat
    df2 = merged_df
    key = "req_id"
    # Merging df1 with a column (value2) from df2 using 'key' as the common column
    df = df1.merge(df2[[key, 'majority_predict_openchat',"uncertainty_openchat",'is_same']], on=key, how='left')
    df = df.rename(columns={'is_same':'correct'})

    ## only include examples that finished with valid response
    df = df[(df['score'] != "undefined") & (df['predict']!= "undefined")]

    df.reset_index(drop=True, inplace=True)

    correct = [0 if value == "yes" else 1 for value in df.predict.values]

    df.correct = correct

    ## variant examples
    variance = []
    for req_id in df['req_id'].unique(): 

        if df[df['req_id'] == req_id]['predict'].nunique() > 1: 
            print(req_id)
            variance.append(req_id)

     # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #req_ids = []
    index_dic = {}
    separated_activations = []
    separated_labels = []
    reqs_order = []
    for req_id in df['req_id'].unique():

        req_df = df[df['req_id'] == req_id].index

        #req_ids.append(req_df)
        index_dic[req_id] = list(req_df)
        
        temp_activations = df[df['req_id'] == req_id].attentions
        activations = np.array([list(sample.values()) for sample in temp_activations.values])#.shape
        batch_length = len(temp_activations)
        dim = 128
        activations = np.reshape(activations, (batch_length, 32, 32, dim))

        temp_labels =[1 if label==True else 0 for label in df[df['req_id'] == req_id]['correct'].values]
        separated_labels.append(temp_labels)
        separated_activations.append(activations)
        reqs_order.append(req_id)
    # create model

    #train_idxs = [reqs_order.index(i) for i in variance]
    train_idxs = np.arange(len(reqs_order))
    #print(train_idxs)
    #train_idxs = np.arange(len(list(index_dic.keys())))
    indexes = np.arange(len(reqs_order))
    # Create a random generator with a specific seed
    seed = 42  # You can choose your own seed value
    rng = np.random.default_rng(seed)
    size = int(len(reqs_order)*(1-args.val_ratio))
    #print(size)
    train_set_idxs = rng.choice(train_idxs, size=size, replace=False)
    val_set_idxs = np.array([x for x in indexes if x not in train_set_idxs])


    tokenizer, model = load_model(args.model_name)

    # define number of layers and heads
    num_layers = 32 #model.config.num_hidden_layers
    num_heads = 32 #model.config.num_attention_heads

    # load activations 
    #head_wise_activations = np.load(f"../features/{args.model_name}_{args.dataset_name}_head_wise.npy")
    #labels = np.load(f"../features/{args.model_name}_{args.dataset_name}_labels.npy")
    #head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    # tuning dataset: no labels used, just to get std of activations along the direction
    #activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
    #tuning_activations = np.load(f"../features/{args.model_name}_{activations_dataset}_head_wise.npy")
    #tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
    #tuning_labels = np.load(f"../features/{args.model_name}_{activations_dataset}_labels.npy")

    #separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)

    #train_idxs = np.arange(len(df))

    # pick a val set using numpy
    #train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
    #val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

    # get directions
    if args.use_center_of_mass:
        com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
    else:
        com_directions = None
    
    top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
        
    print("Heads intervened: ", sorted(top_heads))

    tuning_activations = separated_activations
    tuning_activations = np.concatenate(tuning_activations, axis = 0)
        
    com_directions = None
    interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)

    for head_out_name, list_int_vec in interventions.items():
        layer_no = int(head_out_name.split('.')[2])
        displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))

        for head_no, head_vec, std in list_int_vec:
            displacement[head_no] = args.alpha * std * head_vec
        device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index

        displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
        print(displacement)
        
        print(model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
        bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
        model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)

    save_folder = f"results_dump/{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}"
    if os.path.exists(save_folder):
      shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    model.config.oproj_bias = True
    model.save_pretrained(save_folder, safe_serialization=False, max_shard_size="10GB")
    tokenizer.save_pretrained(save_folder)

if __name__ == "__main__":
    main()
