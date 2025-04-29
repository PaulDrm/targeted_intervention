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

#from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx,

#, get_separated_activations

from ut_intervention_utils import get_interventions_dict, get_top_heads, get_com_directions #, get_fisher_lda_directions
from ut_processing_utils import load_model, select_device
sys.path.append('../app/')

from intervention.reasoning import eval_intervention, eval_intervention_batch, parse_output, extract_final_answer 
import re

def extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset='requirements_data'):

    if dataset== 'requirements_data':
        final_answer, predict = extract_final_answer(output, cot=cot, internal_cot=internal_cot)
    
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
def get_precision_recall_id(df):

    df_openchat = df
    predict_openchat = generate_majority_predictions(df_openchat)

    df_gpt4 = pd.read_json("../results_gpt4_cot_moon_complete.json")
    predict_gpt4 = generate_majority_predictions(df_gpt4)

    # Merge the DataFrames on the 'req_id' column
    merged_df = pd.merge(predict_gpt4, predict_openchat, on='req_id', suffixes=('_gpt4', '_openchat'))

    # Compare the values in the 'majority_predict' column
    merged_df['is_same'] = merged_df['majority_predict_gpt4'] == merged_df['majority_predict_openchat']
    merged_df.head(5)

    df = merged_df
    df = df[df['uncertainty_openchat'] > 0.5]
    df['majority_predict_openchat'] = df['majority_predict_openchat'].apply(lambda x: 1 if x == "yes" else 0)
    df['majority_predict_gpt4'] = df['majority_predict_gpt4'].apply(lambda x: 1 if x == "yes" else 0)

    # precision = precision_score(df['majority_predict_gpt4'], df['majority_predict_openchat'])
    # recall = recall_score(df['majority_predict_gpt4'], df['majority_predict_openchat'])

    #gt = [1 if i == 0 else 0 for i in df['majority_predict_gpt4'] ]
    #pred = [1 if i == 0 else 0 for i in df['majority_predict_openchat'] ]
    precision = precision_score(df['majority_predict_gpt4'], df['majority_predict_openchat'])
    recall = recall_score(df['majority_predict_gpt4'], df['majority_predict_openchat'])

    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    return precision, recall

def get_precision_recall(df):

    df_openchat = df
    predict_openchat = generate_majority_predictions(df_openchat)

    df_gpt4 = pd.read_json("../results_gpt4_cot_moon_complete.json")
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
    df = df[df['uncertainty_openchat'] > 0.5]

    precision = len(df[(df.final_answer == True) & (df.correct == True)]) / (len(df[(df.final_answer == True) & (df.correct == True)]) + len(df[(df.final_answer == True) & (df.correct == False)]))

    recall = len(df[(df.final_answer == True) & (df.correct == True)]) / (len(df[(df.final_answer == True) & (df.correct == True)]) + len(df[(df.final_answer == False) & (df.correct == False)]))

    return precision, recall

import llama
# def load_model(model_name):

#     LOAD_8BIT = False #True
#     BASE_MODEL = model_name

#     tokenizer = llama.LlamaTokenizer.from_pretrained(BASE_MODEL)
#     model = llama.LlamaForCausalLM.from_pretrained(BASE_MODEL, low_cpu_mem_usage=True, load_in_8bit=LOAD_8BIT, torch_dtype=torch.float16, device_map="auto")

#     if "openchat" in model.config._name_or_path:

#         model.generation_config.pad_token_id = 0
#         tokenizer.pad_token_id = 0

#     return tokenizer, model 

import ast

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
    # parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    # parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model', default=False)
    # parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    # parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.5)
    #parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_center_of_mass', type=lambda x: (str(x).lower() == 'true'), help='Whether to use the center of mass or not')
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    #parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--layer', type=int, help='layer for intervention')
    #parser.add_argument('--head', type=int, help='head for intervention')
    parser.add_argument('--input_path', type=str, help='input path')
    parser.add_argument('--output_path', type=str, help='output path')
    parser.add_argument('--dataset_name', type=str, default='requirements_data')
    parser.add_argument('--add_or_subtract', type=lambda x: (str(x).lower() == 'true'), default='true', help='if intervention is added or substract to activations')
    # parser.add_argument('--judge_name', type=str, required=False)
    # parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument("--list_of_heads", type=str, default="", action=ParseListOfLists, help="Input should be a list of lists (e.g., [['11', '0'], ['13', '0']]).")
    parser.add_argument('--prompt_type', type=str, default="ab_cot")
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--consistency_factor', type=int, default=6)

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
        # correct = [0 if value == "yes" else 1 for value in df.predict.values]
        # df.correct = correct

    # set seeds
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

    print(len(separated_activations))

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
    device = select_device(min_vram_gb=20)
    tokenizer, model = load_model(args.model_name, device)

    # Iterate over all folds
    for fold in range(args.num_fold):
        print(f"\nFold {fold + 1}/{args.num_fold}")
        
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
        val_set_idxs = val_idxs
        print(train_set_idxs)
        print(train_set_idxs.shape)
        print(val_set_idxs)
        print(val_set_idxs.shape)
        train_set = df.loc[train_index_expanded]
        val_set = df.loc[val_index_expanded]
        test_set = df.loc[test_index_expanded]

        num_heads =32
        num_layers = 32

        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir, specific_heads=list_of_heads) #(13,0)]) #(14, 1)]) #[(31,14)])
        print("Heads intervened: ", sorted(top_heads))
        tuning_activations = separated_activations
        tuning_activations = np.concatenate(tuning_activations, axis = 0)
        
        # tuning
        # print(tuning_activations.shape)
        # print(len(tuning_activations))
        # print(tuning_activations.shape)

        # get directions
        if args.use_center_of_mass:
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_activations, separated_labels)
            #com_directions = get_fisher_lda_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_activations, separated_labels)
        
        else:
            com_directions = None

        interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)
        def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'): 
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                #print(head)
                #print(direction)
                direction_to_add = torch.tensor(direction).to(head_output.device)
                                    #print(direction_to_add)
                if start_edit_location == 'lt': 
                    head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                    
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
            
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output
        
        def lt_modulated_vector_quantile(head_output, layer_name, start_edit_location='lt'): 
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                #print("quantile")
                # Calculate the threshold for the top 0.5 quantile
                quantile_threshold = np.quantile(np.abs(direction), 0.5)
                #print("Quantile threshold: ", quantile_threshold)
                # Set values whose magnitude is below the threshold to 0
                direction = np.where(np.abs(direction) >= quantile_threshold, direction, 0)

                direction_to_add = torch.tensor(direction).to(head_output.device.index)
                #print(direction_to_add)
                if start_edit_location == 'lt': 
            
                    head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                    
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output
        
        # print(interventions)
        MAX_NEW_TOKENS = 600
        generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
                    "do_sample": True, 
                    "num_beams" : 1, 
                    "num_return_sequences" : 1, 
                    "temperature": args.temperature, #0.8,# 0.8, 
                    "top_p": 0.95,
                #"min_new_tokens": 256, 
                #"no_repeat_ngram_size": 12, 
                #"begin_suppress_tokens": [2], 
                    }
        
        results = []
        counter = 0 

        #if len(test_set.req_id.unique())*10 < test_set.shape[0]:
        # Repeat each row n times
        #n = 4 //  train_set.shape[0]
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

        #for row in tqdm(train_set.iterrows()):
        #for row in tqdm(df.iterrows()):
        # Iterate over the DataFrame in batches
        batch_size = 2
        for start_idx in tqdm(range(0, len(test_set), batch_size)):
            end_idx = min(start_idx + batch_size, len(test_set))
            batch = test_set.iloc[start_idx:end_idx]
        #for row in tqdm(test_set.iterrows()):
            #prompts = batch.prompt.values.tolist()#row[1].prompt
            #print(prompts)
            # if args.dataset_name != "requirements_data" and args.prompt_type == "ab":
            #     prompts = prompt+ " ("
            prompts = batch.prompt.values.tolist()

            #prompts =[prepare_prompt(prompt, tokenizer, args.dataset_name, args.prompt_type) for prompt in prompts] 


            #if counter ==50: 
            #    break

            output = eval_intervention_batch(
            prompts,
            model = model,
            tokenizer = tokenizer,
            stopping_criteria = None,
            device = 'cuda',
            interventions=interventions, 
            intervention_fn= lt_modulated_vector_add if args.add_or_subtract else lt_modulated_vector_quantile, #lt_modulated_vector_add,
            #intervention_fn=lt_modulated_vector_subtract,
            **generation_args, 
            )
           

            # Using list comprehension to process the outputs
            new_outputs = [parse_output(out, prompts[i], tokenizer) for i, out in enumerate(output[0])]
            #print(output)
            for i, new_output in enumerate(new_outputs):

                row = batch.iloc[i]
                prompt = prompts[i]
                
                if args.dataset_name != "requirements_data" and args.prompt_type == "ab":
                    new_output = "(" + new_output
                
                final_answer, predict = extract_final_answer_dataset(new_output, cot=True, internal_cot=False, dataset=args.dataset_name)
                
                if args.dataset_name != "requirements_data" and args.prompt_type != "open_ended":
                    final_answer, predict = extract_final_answer_dataset(new_output, cot=True, internal_cot=False, dataset=args.dataset_name)
                    gt = row['gt'].strip()
                    correct = gt == predict
                    results.append({
                        id_column: row[id_column], 
                        "prompt": prompt, 
                        "output": new_output, 
                        "final_answer": final_answer, 
                        "gt": gt, 
                        "predict": correct,
                        "heads": top_heads, 
                        #"score": score
                    })
                
                elif args.dataset_name == "requirements_data":
                    final_answer, predict = extract_final_answer_dataset(new_output, cot=True, internal_cot=False, dataset=args.dataset_name)
                    gt = row['final_answer'] if row['correct'] else not row['final_answer']
                    results.append({
                        id_column: row[id_column], 
                        "prompt": prompt, 
                        "output": new_output, 
                        "final_answer": final_answer, 
                        "predict": predict, 
                        "gt": gt,
                        "heads": top_heads,
                        "seed": args.seed
                        #"score": score, 
                        
                    })
                
                else: 
                    results.append({
                        id_column: row[id_column], 
                        "question": row['question'], 
                        "prompt": prompt, 
                        "output": new_output, 
                        "answer": new_output, 
                        "heads": top_heads, 
                        # "score": score,
                    })
    
                counter += 1

        curr_fold_results = pd.DataFrame(results)

        head_string = ""
        for head in top_heads:#list_of_heads:
            head_string = head_string + str(head[0]) + "_"+ str(head[1]) + "_"
        
        if args.num_heads == 1:
            curr_fold_results.to_json(f"{args.output_path}/results_intervention_{int(args.alpha)}_head_{head_string}_seed_{args.seed}_consistency_{args.consistency_factor}_fold_{fold}.json", orient='records', indent=4)
        else:
            curr_fold_results.to_json(f"{args.output_path}/results_intervention_{int(args.alpha)}_number_heads_{len(top_heads)}_seed_{args.seed}_consistency_{args.consistency_factor}_fold_{fold}.json", orient='records', indent=4)
        
        #print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}")
        if args.prompt_type != "open_ended":
            print(curr_fold_results.predict.value_counts())

            with open(f'{args.output_path}/overall_results.txt', 'a') as f:
                # Redirect the print output to the file
                # print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}", file=f)
                print(f"For dataset fold {i} and alpha {args.alpha} and heads {head_string} and seed {args.seed} and consistency {args.consistency_factor}", file=f)
                print(curr_fold_results.predict.value_counts(),file = f)
                print(curr_fold_results.final_answer.value_counts(),file = f)

            results = []
            counter = 0 
            
            value_counts = curr_fold_results.predict.value_counts().to_dict()
            value_counts['alpha'] = args.alpha
            value_counts['heads'] = top_heads #.tolist() #args.list_of_heads
            value_counts['seed'] = args.seed
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

            
        # #print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}")
        # print(curr_fold_results.predict.value_counts())
        # with open(f'{args.output_path}/overall_results.txt', 'a') as f:
        #     # Redirect the print output to the file
        #     #print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}", file=f)
        #     print(f"For dataset fold {i} and alpha {args.alpha} and heads {args.layer} {args.head}", file=f)
        #     print(curr_fold_results.predict.value_counts(),file = f)
        #     print(curr_fold_results.final_answer.value_counts(),file = f)

        # results = []
        # counter = 0 
        
        # value_counts = curr_fold_results.predict.value_counts().to_dict()
        # value_counts['alpha'] = args.alpha
        # value_counts['layer'] = args.layer
        # value_counts['head'] = args.head

        # # Path to your JSON file
        # json_file_path = f'{args.output_path}/overall_results.json'
        # # Load existing data from the JSON file
        # try:
        #     with open(json_file_path, 'r') as file:
        #         data = json.load(file)
        # except FileNotFoundError:
        #     data = []

        # # Add the new data to the existing list (or create a new list if the file was not found)
        # data.append(value_counts)

        # # Write the updated data back to the JSON file
        # with open(json_file_path, 'w') as file:
        #     json.dump(data, file, indent=4)
        
if __name__ == "__main__":
    main()   
    