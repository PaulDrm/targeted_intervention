import torch
import numpy as np
from einops import rearrange
import pickle
import os
from tqdm import tqdm
import pandas as pd
import ast 
import argparse
from datasets import load_dataset
import json
import sys
sys.path.append('../')
from src.old.utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, load_model, prepare_prompt, run_llama_intervention, run_llama_intervention_batch

sys.path.append('../app/')
from intervention.reasoning import eval_intervention, parse_output, extract_final_answer, evaluate 
import re

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

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', help='model name')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.5)
    #parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_center_of_mass', type=lambda x: (str(x).lower() == 'true'), help='Whether to use the center of mass or not')
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    #parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument("--list_of_heads", type=str, default="", action=ParseListOfLists, help="Input should be a list of lists (e.g., [['11', '0'], ['13', '0']]).")
    parser.add_argument('--input_path', type=str, help='input path')
    parser.add_argument('--output_path', type=str, help='output path')
    parser.add_argument('--dataset_name', type=str, default='requirements_data')
    parser.add_argument('--add_or_subtract', type=lambda x: (str(x).lower() == 'true'), default='true', help='if intervention is added or substract to activations')
    parser.add_argument('--test_set_input_path', type=str)
    parser.add_argument('--prompt_type', type=str, default="open_ended")
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--consistency_factor', type=int, default=6)

    #parser.add_argument('--normalize_with_activations', type=lambda x: (str(x).lower() == 'true'), default='true')

    args = parser.parse_args()

    if not os.path.exists(f"{args.output_path}"):
        os.mkdir(f"{args.output_path}")

    # Print all arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    if args.list_of_heads != "":
        list_of_heads = [[int(head[0]), int(head[1])] for head in args.list_of_heads]
        print("Parsed list of lists:", list_of_heads)
    
    else:
        list_of_heads = None

    df = pd.read_json(args.input_path)
   
    if args.dataset_name != "requirements_data":
        id_column = "data_id"
    else: 
        id_column = "req_id"
        #correct = [0 if value == "yes" else 1 for value in df.predict.values]
        #df.correct = correct

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #req_ids = []
    index_dic = {}
    separated_activations = []
    separated_labels = []
    reqs_order = []
    ## TODO Make more visible 
    use_attention = True
    if use_attention:
        column = "attentions" 
    else: 
        column =  'o_proj_activations'
    
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

    # get folds using numpy
    #print("Number of folds: ", args.num_fold)
    #fold_idxs = np.array_split(np.arange(len(list(index_dic.keys()))), args.num_fold)
    #print(fold_idxs)

    number_of_examples = np.arange(len(reqs_order))    
    fold_idxs = np.array_split(number_of_examples, args.num_fold)

    tokenizer, model = load_model(args.model_name, device = "cuda:1")

    for i in range(args.num_fold):
        #if i == 0:
        #    continue
        if args.num_fold == 1: 
            train_idxs = np.arange(len(reqs_order))

        else: 
            train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        
        seed = 42  # You can choose your own seed value
        rng = np.random.default_rng(seed)
        size = int(len(train_idxs)*(1-args.val_ratio))
        #print(size)
        train_set_idxs = rng.choice(train_idxs, size=size, replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        if len(fold_idxs) == 1:
            test_idxs = val_set_idxs
        else:
            test_idxs = fold_idxs[i]

        print(len(fold_idxs))
        #print(fold_idxs))
        print(len(train_idxs))
        print(len(train_set_idxs))
        print(len(val_set_idxs))
        print(len(test_idxs))
        print("Test indexes:", test_idxs)
        # pick a val set using numpy
        #train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        #val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
        len(val_set_idxs)

        train_index_list = np.concatenate([list(index_dic.values())[i] for i in train_set_idxs], axis = 0)
        train_set = df.loc[train_index_list]

        if len(val_set_idxs) > 0:
            val_index_list = np.concatenate([list(index_dic.values())[i] for i in val_set_idxs], axis = 0)
            val_set = df.loc[val_index_list]
        
        else:
            val_set_idxs = train_set_idxs

        #print(val_set_idxs[0])
        if len(test_idxs) > 0:
            test_index_list = np.concatenate([list(index_dic.values())[i] for i in test_idxs], axis = 0)
            #print()
            test_set = df.loc[test_index_list]

        # df_gt = pd.read_json("../datasets/refusal/dataset_gt.json")
        # filt = df_gt[df_gt.predict == False].data_id
        
        # negative_samples = test_set[test_set['data_id'].isin(filt)]
        # positive_samples = test_set[~test_set['data_id'].isin(filt)]
        # test_set = pd.concat([positive_samples.iloc[0:26], negative_samples.iloc[0:26]])#.head()
        num_heads =32
        num_layers = 32

        if args.use_center_of_mass: 
            top_heads = list_of_heads
            probes = []
        else:
            #top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir, specific_heads=[(args.layer,args.head)]) #(13,0)]) #(14, 1)]) #[(31,14)])
            top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir, specific_heads=list_of_heads)#[(args.layer,args.head)]) #(13,0)]) #(14, 1)]) #[(31,14)])
        
        
        print("Heads intervened: ", sorted(top_heads))
        print("Number of heads intervened: ", len(top_heads))
        # print("Type of top heads: ", type(top_heads))
        # print("Type of one head: ", type(top_heads[0]))
        # print("Type of one entry in head: ", type(top_heads[0][0]))
        top_heads = [[int(item) for item in tup] for tup in top_heads]
        print("Top heads: ", top_heads)
        tuning_activations = separated_activations
        tuning_activations = np.concatenate(tuning_activations, axis = 0)
        
        #tuning
        #print(tuning_activations.shape)
        #print(len(tuning_activations))
        #print(tuning_activations.shape)

        # get directions
        if args.use_center_of_mass:
            print("Using center of mass...")
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_activations, separated_labels)
        else:
            com_directions = None

        interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)
        
        def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'): 
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                direction_to_add = torch.tensor(direction).to(head_output.device)
                
                #print(direction_to_add)
                if start_edit_location == 'lt': 
                    head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                    
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output
        
        def lt_modulated_vector_subtract(head_output, layer_name, start_edit_location='lt'): 
                head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
                for head, direction, proj_val_std in interventions[layer_name]:
                    
                    direction_to_add = torch.tensor(direction).to(head_output.device.index)
                    if start_edit_location == 'lt': 
                
                        head_output[:, -1, head, :] -= args.alpha * proj_val_std * direction_to_add
                        
                    else: 
                        head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
                head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                return head_output
        

        # from get_activations import load_model

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

        results = run_llama_intervention(args, tokenizer, model, interventions, test_set)

        curr_fold_results = pd.DataFrame(results)

        # if args.dataset_name == "requirements_data":
        #     correct = [0 if value == "yes" else 1 for value in df.predict.values]
        #     curr_fold_results.correct = correct
            #print(curr_fold_results.head(3))
        head_string = ""
        for head in top_heads:#list_of_heads:
            head_string = head_string + str(head[0]) + "_"+ str(head[1]) + "_"
        
        curr_fold_results.to_json(f"{args.output_path}/results_intervention_{int(args.alpha)}_number_heads_{len(top_heads)}.json", orient='records', indent=4)
            
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
            value_counts['alpha'] = args.alpha
            value_counts['heads'] = top_heads #.tolist() #args.list_of_heads

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
    