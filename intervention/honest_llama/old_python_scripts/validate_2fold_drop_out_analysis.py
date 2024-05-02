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

import sys
sys.path.append('../')
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions

sys.path.append('../app/')
from reasoning import eval_intervention, parse_output, extract_final_answer 

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
def load_model(model_name):

    LOAD_8BIT = False #True
    BASE_MODEL = model_name

    tokenizer = llama.LlamaTokenizer.from_pretrained(BASE_MODEL)
    model = llama.LlamaForCausalLM.from_pretrained(BASE_MODEL, low_cpu_mem_usage=True, load_in_8bit=LOAD_8BIT, torch_dtype=torch.float16, device_map="auto")

    if "openchat" in model.config._name_or_path:

        model.generation_config.pad_token_id = 0

    return tokenizer, model 


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
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    #parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--layer', type=int, help='layer for intervention')
    parser.add_argument('--head', type=int, help='head for intervention')
   
    # parser.add_argument('--judge_name', type=str, required=False)
    # parser.add_argument('--info_name', type=str, required=False)
    args = parser.parse_args()

    df = pd.read_json("../requirements_data/dataframe_open_chat_cot_moon_06022024_attentions_gt.json")
    #df = pd.read_json("../requirements_data/dataframe_open_chat_cot_moon_18012024_attentions_gt.json")

    correct = [0 if value == "yes" else 1 for value in df.predict.values]

    df.correct = correct

    for i in range(5): 
        print("Label: ", df.correct.iloc[i])
        print("Prediction: ", df.predict.iloc[i])

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

    # get folds using numpy
    #print("Number of folds: ", args.num_fold)
    #fold_idxs = np.array_split(np.arange(len(list(index_dic.keys()))), args.num_fold)
    #print(fold_idxs)

    number_of_examples = np.arange(len(reqs_order))    
    fold_idxs = np.array_split(number_of_examples, args.num_fold)

    tokenizer, model = load_model(args.model_name)

    for i in range(args.num_fold):
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

        val_index_list = np.concatenate([list(index_dic.values())[i] for i in val_set_idxs], axis = 0)
        val_set = df.loc[val_index_list]

        train_index_list = np.concatenate([list(index_dic.values())[i] for i in train_set_idxs], axis = 0)
        train_set = df.loc[train_index_list]
        #print(val_set_idxs[0])
        test_index_list = np.concatenate([list(index_dic.values())[i] for i in test_idxs], axis = 0)
        #print()
        test_set = df.loc[test_index_list]

        num_heads =32
        num_layers = 32

        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir, specific_heads=[(args.layer,args.head)]) #(13,0)]) #(14, 1)]) #[(31,14)])
        
        print("Heads intervened: ", sorted(top_heads))

        tuning_activations = separated_activations
        tuning_activations = np.concatenate(tuning_activations, axis = 0)
        #tuning
        #print(tuning_activations.shape)
        #print(len(tuning_activations))
        #print(tuning_activations.shape)

        com_directions = None
        interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)

        def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'): 
                head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
                for head, direction, proj_val_std in interventions[layer_name]:
                    #print(head)
                    #print(direction)
                    direction_to_add = torch.tensor(direction).to(head_output.device.index)
                    #print(direction_to_add)
                    if start_edit_location == 'lt': 
                
                        head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                        
                    else: 
                        head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
                head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                return head_output
        
        def lt_modulated_vector_drop_out(head_output, layer_name, start_edit_location='lt'): 
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                #print(head)
                #print(direction)
                direction_to_add = torch.tensor(direction).to(head_output.device.index)
                #print(direction_to_add)
                if start_edit_location == 'lt': 
            
                    #head_output[:, -1, head, :] == args.alpha * proj_val_std * direction_to_add
                    zero_vector = torch.zeros_like(head_output[:, -1, head, :])
                    #head_output[:, -1, head, :] = 0.0 #zero_vector
                    #print(head_output[:, -1, head, :][0:5])
                    
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add

            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output
        
        def lt_pertubated_vector(head_output, layer_name, start_edit_location='lt'): 

            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                #print(head)
                #print(direction)
                direction_to_add = torch.tensor(direction).to(head_output.device.index)
                #print(direction_to_add)
                if start_edit_location == 'lt': 
            
                    #head_output[:, -1, head, :] == args.alpha * proj_val_std * direction_to_add
                    zero_vector = torch.zeros_like(head_output[:, -1, head, :])
                    #head_output[:, -1, head, :] = 0.0 #zero_vector
                    #head_output[:, -1, head, :] = 0.0 #zero_vector
                    #print(head_output[:, -1, head, :][0:5])
                    
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add

            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output
        
        # print(interventions)
        # from get_activations import load_model

        MAX_NEW_TOKENS = 600
        generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
                    "do_sample": True, 
                    "num_beams" : 1, 
                    "num_return_sequences" : 1, 
                    "temperature": 0.8,# 0.8, 
                    "top_p": 0.95,
                #  "min_new_tokens": 256, 
                #"no_repeat_ngram_size": 12, 
                #  "begin_suppress_tokens": [2], 
                    }
        
        results = []
        counter = 0 

        for row in tqdm(train_set.iterrows()):
        #for row in tqdm(df.iterrows()):
            prompt = row[1].prompt
            #
            if counter == 60: 
                break
                #print(prompt[-100:])
            output = eval_intervention(
            row[1].prompt,
            model = model,
            tokenizer = tokenizer,
            stopping_criteria = None,
            device = 'cuda',
            interventions=interventions, 
            intervention_fn=lt_modulated_vector_drop_out,
            **generation_args, 
            )

            scores= torch.softmax(output[1].scores[-1],1)
            score = round(torch.max(scores).item(),2)#.item()

            output = parse_output(output[0], prompt, tokenizer)
            #print(output)
            final_answer, predict = extract_final_answer(output, cot=True, internal_cot=False)

            results.append({"req_id": row[1].req_id, "prompt": prompt, "output": output, "final_answer": final_answer, "predict": predict, "score": score}),
            
            counter += 1
        curr_fold_results = pd.DataFrame(results)

        #print(curr_fold_results.head(3))
        
        curr_fold_results.to_json(f"../intervention_results/results_test_openchat_intervention_no_train_{int(args.alpha)}_{args.layer}_{args.head}_fold_{i}.json", orient='records', indent=4)

        try:
            precision, recall = get_precision_recall(curr_fold_results)
        except:
            precision = 0
            recall = 0
            
        print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}")
        print(curr_fold_results.predict.value_counts())
        with open('../intervention_results/overall_results.txt', 'a') as f:
            # Redirect the print output to the file
            print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}", file=f)
            print(curr_fold_results.predict.value_counts(),file = f)

        results = []
        counter = 0 

        # df = pd.read_json("../requirements_data/dataframe_open_chat_cot_moon_06022024_attentions_gt.json")
        # correct = [0 if value == "yes" else 1 for value in df.predict.values]
        # df.correct = correct

        # test_set = df[~df.req_id.isin(train_set.req_id.unique())]


        # for row in tqdm(test_set.iterrows()):
        # #for row in tqdm(df.iterrows()):
        #     prompt = row[1].prompt
        #     #
        #     if counter == 0: 
        #         print(prompt[-100:])
        #     output = eval_intervention(
        #     row[1].prompt,
        #     model = model,
        #     tokenizer = tokenizer,
        #     stopping_criteria = None,
        #     device = 'cuda',
        #     interventions=interventions, 
        #     intervention_fn=lt_modulated_vector_add,
        #     **generation_args, 
        #     )

        #     scores= torch.softmax(output[1].scores[-1],1)
        #     score = round(torch.max(scores).item(),2)#.item()

        #     output = parse_output(output[0], prompt, tokenizer)
        #     #print(output)
        #     final_answer, predict = extract_final_answer(output, cot=True, internal_cot=False)
        #     results.append({"req_id": row[1].req_id, "prompt": prompt, "output": output, "final_answer": final_answer, "predict": predict, "score": score})
        #     counter += 1
        # curr_fold_results = pd.DataFrame(results)

        # #print(curr_fold_results.head(3))
        
        # curr_fold_results.to_json(f"../intervention_results/results_test_openchat_intervention_no_valid_{int(args.alpha)}_{args.layer}_{args.head}_fold_{i}.json", orient='records', indent=4)

        # try:
        #     precision, recall = get_precision_recall(curr_fold_results)
        # except:
        #     precision = 0
        #     recall = 0
            
        # print(f"Valid data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and {args.layer}_{args.head}")
        # print(curr_fold_results.predict.value_counts())
        # with open('../intervention_results/overall_results.txt', 'a') as f:
        #     # Redirect the print output to the file
        #     print(f"Valid data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and {args.layer}_{args.head}", file=f)
        #     print(curr_fold_results.predict.value_counts(),file = f)
        
if __name__ == "__main__":
    main()   
    