import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset

import json 

import sys
sys.path.append('../')
from src.old.utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, load_model, prepare_prompt, prepare_prompt, run_llama_intervention, run_llama_intervention_batch

import time
import statistics
sys.path.append('../app/')
from intervention.reasoning import eval_intervention, parse_output, extract_final_answer 

def generate_majority_predictions(df): 
    predict = {}
    predict = []
    for req_id in df['req_id'].unique(): 

        req_df = df[df['req_id'] == req_id]
        maj_ele = req_df['predict'].value_counts().index[0]
        uncertainty = max(req_df['predict'].value_counts()) / len(req_df)
        #print(req_id)
        mean_score = req_df[(req_df['req_id']==req_id)& (req_df['predict']!= "undefined") & (req_df['score']!="undefined")].score.mean()
        #predict[req_id] = {"majority_predict" : maj_ele, "uncertainty" : uncertainty}
        predict.append({"req_id": req_id, "majority_predict" : maj_ele, "uncertainty" : uncertainty, "mean_score": mean_score})

    predict = pd.DataFrame(predict)
    return predict

from sklearn.metrics import recall_score, precision_score

def extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset='requirements_data'):

    if dataset== 'requirements_data':
        final_answer, predict = extract_final_answer(output, cot=cot, internal_cot=internal_cot)
    
    elif dataset == "golden_gate":

        if any([s in output.lower() for s in ["golden gate"]]):

            final_answer ="(A)"
            predict = "(A)"
        else:
            final_answer = "(B)"
            predict = "(B)"

    else: 

        matched_text = output
        #print(matched_text)
        if "(a)" in matched_text.lower():
            final_answer = "(A)"
            
        elif "(b)" in matched_text.lower():
            final_answer = "(B)"
        
        else:
            final_answer = "undefined"  

        if final_answer == "(A)": 
            predict = "(A)" 

        elif final_answer == "undefined":
            predict = "undefined"   
        else:
            predict = "(B)"

    return final_answer, predict

import llama
from transformers import AutoTokenizer

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
    #parser.add_argument('--layer', type=int, help='layer for intervention')
    #parser.add_argument('--head', type=int, help='head for intervention')
    parser.add_argument('--input_path', type=str, help='input path')
    parser.add_argument('--output_path', type=str, help='output path')
    parser.add_argument('--dataset_name', type=str, default='requirements_data')
    parser.add_argument('--add_or_subtract', type=lambda x: (str(x).lower() == 'true'), default='true', help='if intervention is added or substract to activations')
    parser.add_argument('--test_set_input_path', type=str)
    parser.add_argument('--prompt_type', type=str, default="open_ended")
    parser.add_argument('--add_proj_val_std', type=lambda x: (str(x).lower() == 'true'), default='true')
    parser.add_argument('--temperature', type=float, default=0.8)
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
    
    tokenizer, model = load_model(args.model_name, device="cuda:0")

        #if i == 0:
        #    continue
        #if i == 1:
        #    continue
        # #train_idxs = [reqs_order.index(i) for i in variance]
        # 
        # #print(train_idxs)
        # #train_idxs = np.arange(len(list(index_dic.keys())))
        # indexes = np.arange(len(reqs_order))
        # # Create a random generator with a specific seed
        # seed = 42  # You can choose your own seed value
        # rng = np.random.default_rng(seed)
        # size = int(len(reqs_order)*(1-args.val_ratio))
        # #print(size)
        # train_set_idxs = rng.choice(train_idxs, size=size, replace=False)
        # val_set_idxs = np.array([x for x in indexes if x not in train_set_idxs])
        
    train_idxs = np.arange(len(reqs_order))
    seed = 42  # You can choose your own seed value
    rng = np.random.default_rng(seed)
    size = int(len(train_idxs)*(1-args.val_ratio))
    #print(size)
    train_set_idxs = rng.choice(train_idxs, size=size, replace=False)
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

    test_idxs = val_set_idxs
            
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

    num_heads =32
    num_layers = 32
    if not args.use_center_of_mass:
        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir, specific_heads=None) #(13,0)]) #(14, 1)]) #[(31,14)])
            
    else: 
        probes = []

    durations = []  # List to store the duration of each iteration
    threshold = 13
    for layer in range(8, num_layers):
        #if layer > 20: 
        #a    break
        for h in range(num_heads):

            print(f"Layer: {layer}, Head: {h}")

            args.layer = layer
            args.head = h

            #top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir, specific_heads=[(args.layer, args.head)]) #(13,0)]) #(14, 1)]) #[(31,14)])
            top_heads = [(args.layer, args.head)]
            print("Heads intervened: ", sorted(top_heads))

            tuning_activations = separated_activations
            tuning_activations = np.concatenate(tuning_activations, axis = 0)
            #print(tuning_activations.shape)
            #print(len(tuning_activations))
            #print(tuning_activations.shape)
            # get directions
            if args.use_center_of_mass:
                com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_activations, separated_labels)
            else:
                com_directions = None
            interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)

            def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt', add_proj_val_std = args.add_proj_val_std ): 
 
                    head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
                    for head, direction, proj_val_std in interventions[layer_name]:
                        #print(head)
                        #print(direction)
                        direction_to_add = torch.tensor(direction).to(head_output.device.index)
                        #print(direction_to_add)
                        if start_edit_location == 'lt': 
                            if add_proj_val_std: 
                                head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add  
                            else: 
                                head_output[:, -1, head, :] += args.alpha * direction_to_add
                        else: 
                            head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
                    
                    head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                    return head_output
            
            def lt_modulated_vector_subtract(head_output, layer_name, start_edit_location='lt'): 
                    head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
                    for head, direction, proj_val_std in interventions[layer_name]:
                        #print(head)
                        #print(direction)
                        direction_to_add = torch.tensor(direction).to(head_output.device.index)
                        #print(direction_to_add)
                        if start_edit_location == 'lt': 
                    
                            head_output[:, -1, head, :] -= args.alpha * proj_val_std * direction_to_add
                            
                        else: 
                            head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
                    head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                    return head_output
            
            MAX_NEW_TOKENS = 600
            generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
                        "do_sample": True, 
                        "num_beams" : 1, 
                        "num_return_sequences" : 1, 
                        "temperature": args.temperature, #0.8,# 0.8, 
                        "top_p": 0.95,
                    #  "min_new_tokens": 256, 
                    #"no_repeat_ngram_size": 12, 
                    #  "begin_suppress_tokens": [2], 
                        }
            
            results = []
            counter = 0 

            if train_set.shape[0] < 4:
                # Repeat each row n times
                n = 4 //  train_set.shape[0]
                train_set = train_set.loc[train_set.index.repeat(n)]
                print(train_set.shape)

            for row in tqdm(train_set.iterrows()):
            #for row in tqdm(df.iterrows()):
                start_time = time.time()  # Start time of the iteration

                prompt = prepare_prompt(row[1].prompt, tokenizer, args.dataset_name, args.prompt_type)
                
                #prompt = row[1].prompt
                #
                if args.dataset_name != "requirements_data" and args.prompt_type == "ab":
                    prompt = prompt+ " ("

                if counter == 30: 
                    break
                try:
                    output = eval_intervention(
                        prompt, 
                        model=model,
                        tokenizer=tokenizer,
                        stopping_criteria=None,
                        device='cuda',
                        interventions=interventions,
                        intervention_fn=lt_modulated_vector_add,  # or lt_modulated_vector_subtract
                        **generation_args,
                    )
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
                 
                scores= torch.softmax(output[1].scores[-1],1)
                score = round(torch.max(scores).item(),2)#.item()

                output = parse_output(output[0], prompt, tokenizer)
                #if args.dataset_name != "requirements_data" and args.prompt_type == "ab":
                #output = "(" +output
            
                #### --> 
                final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
                if args.dataset_name != "requirements_data" and args.prompt_type != "open_ended":
                    final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
                    gt = row[1]['gt']
                    gt = gt.strip()
                    correct = gt == predict
                    results.append({id_column: row[1][id_column], "prompt": prompt, "output": output, "final_answer": final_answer,"gt": gt, "predict": correct, "score": score, }),
                
                elif args.dataset_name == "requirements_data":
                    final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
                    results.append({id_column: row[1][id_column], "prompt": prompt, "output": output, "final_answer": final_answer, "predict": predict, "score": score }),
                
                else: 
                    results.append({id_column: row[1][id_column], "question": row[1]['question'], "prompt": prompt, "output": output, "answer": output, "score": score }),
                
                counter += 1

                end_time = time.time()  # End time of the iteration
                duration = end_time - start_time
                if duration > threshold:
                    print(f"Head {h} took {duration:.4f} seconds, which is above the threshold of {threshold:.4f} seconds.")
                    print("Breaking the loop due to outlier duration.")
                    break

            if len(results)== 0:
                continue
            
            # results = run_llama_intervention_batch(args, tokenizer, model, interventions, test_set)

            curr_fold_results = pd.DataFrame(results)

            curr_fold_results.to_json(f"{args.output_path}/results_test_openchat_intervention_no_train_{int(args.alpha)}_{args.layer}_{args.head}.json", orient='records', indent=4)

            #print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}")
            if args.prompt_type != "open_ended":
                print(curr_fold_results.predict.value_counts())
                with open(f'{args.output_path}/overall_results.txt', 'a') as f:
                    # Redirect the print output to the file
                    #print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}", file=f)
                    print(f"For dataset alpha {args.alpha} and head {args.layer} {args.head}", file=f)
                    print(curr_fold_results.predict.value_counts(),file = f)
                    print(curr_fold_results.final_answer.value_counts(),file = f)

                results = []
                counter = 0 
                
                value_counts = curr_fold_results.predict.value_counts().to_dict()
                value_counts['alpha'] = args.alpha
                value_counts['layer'] = args.layer
                value_counts['head'] = args.head

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
    