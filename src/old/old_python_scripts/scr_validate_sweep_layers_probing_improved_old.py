import torch
import numpy as np
import os
import pandas as pd
import numpy as np
import argparse
import json 
import sys
sys.path.append('../')
#from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, load_model, prepare_prompt, prepare_prompt, run_llama_intervention, run_llama_intervention_batch

from ut_intervention_utils import run_llama_intervention_batch_parallel, get_interventions_dict, get_top_heads, get_com_directions

from ut_processing_utils import load_model, prepare_test_set

from ut_evaluation_utils import evaluate_configurations

sys.path.append('../app/')

from honest_llama.old_python_scripts.run_experiments import process_data

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

    seeds = [5678, 9012]  # List of seeds to test

    #1234, 
    tokenizer, model = load_model(args.model_name)

    for seed in seeds:
        
        args.seed = seed

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # Process the data
        index_dic, separated_activations, separated_labels, reqs_order = process_data(df, id_column)
        
        train_idxs = np.arange(len(reqs_order))
        #seed = 42  # You can choose your own seed value
        rng = np.random.default_rng(args.seed)
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
        
        train_index_list = np.concatenate([list(index_dic.values())[i] for i in train_set_idxs], axis = 0)
        train_set = df.loc[train_index_list]

        val_set_idxs = train_set_idxs
        
        if len(val_set_idxs) > 0:
            val_index_list = np.concatenate([list(index_dic.values())[i] for i in val_set_idxs], axis = 0)
            val_set = df.loc[val_index_list]
            
        #print(val_set_idxs[0])
        if len(test_idxs) > 0:
            test_index_list = np.concatenate([list(index_dic.values())[i] for i in test_idxs], axis = 0)
            test_set = df.loc[test_index_list]
            print(test_set.shape)

        
        args.consistency_factor = 3
        test_set = prepare_test_set(test_set, args)
        print(test_set.shape)

        # if test_set.shape[0] < 4:
            # # Repeat each row n times
            # n = 4 // test_set.shape[0]
            # test_set = test_set.loc[test_set.index.repeat(n)]
            # print(test_set.shape)

        num_heads =32
        num_layers = 32
        if not args.use_center_of_mass:

            top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir, specific_heads=None) #(13,0)]) #(14, 1)]) #[(31,14)])

        else: 
            probes = []

        for layer in range(0, num_layers):
            #if layer > 20: 
            #    break
            #for h in range(num_heads):
            args.layer = layer
            top_heads = [(args.layer, head) for head in range(0,32)]
            
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
            

            results = []
            
            #results = run_llama_intervention(args, tokenizer, model, interventions, test_set)
            results = run_llama_intervention_batch_parallel(args, tokenizer, model, interventions, test_set)
            curr_fold_results = pd.DataFrame(results)

            curr_fold_results.to_json(f"{args.output_path}/results_test_intervention_seed_{int(args.seed)}_{int(args.alpha)}_{args.layer}.json", orient='records', indent=4)

            print(curr_fold_results.predict.value_counts())
            with open(f'{args.output_path}/overall_results.txt', 'a') as f:
                # Redirect the print output to the file
                #print(f"Train data: Precision: {precision}, Recall: {recall} for fold {i} and alpha {args.alpha} and head {args.layer} {args.head}", file=f)
                print(f"For dataset alpha {args.alpha} and {args.layer}", file=f)
                print(curr_fold_results.predict.value_counts(),file = f)
                print(curr_fold_results.final_answer.value_counts(),file = f)

            results = []
            
            value_counts = curr_fold_results.predict.value_counts().to_dict()
            value_counts['alpha'] = args.alpha
            value_counts['layer'] = args.layer
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

if __name__ == "__main__":
    main()   
    