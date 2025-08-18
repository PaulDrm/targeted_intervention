import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm
from functools import partial

import ast 

from utils.ut_processing_utils import (
    load_model,
    layer_head_to_flattened_idx,
    get_attention_activations,
    get_fold_indices,
    extract_final_answer_dataset,
    select_device
    # generate_input_output_prompt # Not directly used in the main script flow provided, but might be in deeper utils
)
from utils.ut_intervention_utils import get_com_directions, lt_modulated_vector_add, get_interventions_dict_variable_alpha
from utils.ut_run_llms import run_llama_intervention_batch

from utils.ut_evaluation_utils import evaluate_configuration_general, append_to_log_file, extract_answer_compare_gt, save_results

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run attention head sweep and evaluation.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Name of the model to load.")
    parser.add_argument("--initial_dataset_path", type=str, required=True, help="Path to the JSON dataset for calculating initial directions and sweep evaluation.")
    parser.add_argument("--evaluation_dataset_path", type=str, required=True, help="Path to the JSON dataset for final evaluation with the best head.")
    parser.add_argument("--layers_to_sweep", type=str, required=True, help="Comma-separated list of layer indices to sweep (e.g., '10,11,12').")
    parser.add_argument("--heads_to_sweep_per_layer", type=str, default=None, help="Optional: Comma-separated list of head indices to sweep for each layer (e.g., '0,1,5'). If None, sweeps all heads.")
    parser.add_argument("--alpha", type=str, default="3.0", help="Alpha value for intervention strength.")
    parser.add_argument("--alphas", type=str, default="[3]", help="Alpha values as a JSON array")
    parser.add_argument("--max_new_tokens", type=int, default=600, help="Max new tokens for generation.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation during final evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results.")
    parser.add_argument("--use_center_of_mass", action="store_true", default=True, help="Use center of mass for directions.")
    parser.add_argument("--use_random_dir", action="store_true", default=False, help="Use random directions.")
    parser.add_argument("--id_column", type=str, default="data_id", help="Column name for data IDs in dataframes.")
    parser.add_argument("--activations_column", type=str, default="activations", help="Column name for activations in dataframes.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for running interventions.")
    parser.add_argument("--consistency_factor", type=int, default=1, help="Consistency factor for preparing test set.")
    parser.add_argument("--dataset_name", type=str, default="survival_instinct", help="Dataset name hint for extract_final_answer_dataset.")
    parser.add_argument("--prompt_type", type=str, default="ab_cot", help="Prompt type hint for extract_final_answer_dataset.")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory.")
    parser.add_argument("--num_fold", type=int, default=1, help="Number of folds for cross-validation.")
    parser.add_argument("--val_ratio", type=float, default=0, help="Ratio of validation set size to development set size.")

    args = parser.parse_args()
    
    return args

def load_and_preprocess_initial_data(data_path, tokenizer, model, args):
    print(f"Loading initial data from: {data_path}")
    df = pd.read_json(data_path)
    # The notebook uses get_attention_activations which also computes 'complete_inputs'
    # We'll assume get_activations_bau is a function known to get_attention_activations
    # or handle it inside get_attention_activations if it's a fixed part of its logic.
    # For now, passing None and assuming get_attention_activations can handle it or has a default.
    df = get_attention_activations(df, tokenizer, model) # Adapt get_activations_bau_func as needed
    print(f"Initial data loaded and preprocessed. Shape: {df.shape}")
    print(df['complete_inputs'].values[0])
    #print(df['activations'].values[0])
    return df

def get_directions(model, df, args=None, id_column = "data_id", column = "activations"):
    fold_index = 0
    #### Purpose of this script is to process attention head activations from a dataframe
    #### So for every unique data_id, it separates the attention head activations into different batches and labels
    index_dic = {}

    ## List of lists with attention head activations for each data_id examples
    separated_activations = []

    ## List of lists with labels for each data_id examples
    separated_labels = []
    data_ids_order = []

    num_layers = model.config.num_hidden_layers 
    num_heads = model.config.num_attention_heads

    if hasattr(model.config, "head_dim"):
        head_dim = model.config.head_dim
    else:
        head_dim = model.config.hidden_size // num_heads

    for data_id in df[id_column].unique():

        ## Necessary? --> used later when expanding train idxs and used for verbose logging
        example_indexes = df[df[id_column] == data_id].index
        ## Gives indexes for samples in the whole dataset
        index_dic[data_id] = list(example_indexes)
        ## Example: {'304_a': [0, 2], '304_b': [1, 3], '294_a': [4, 6], '294_b': [5, 7]} --> Dataset with 4 unique ids with 2 examples each
        
        temp_activations = df[df[id_column] == data_id][column]
        activations = np.array([list(sample.values()) for sample in temp_activations.values]) # [num_examples, num_layers x num_heads, head_dim]
        
        ## Number of example for the current data_id
        number_examples = len(temp_activations)
        
        ## split into attention heads
        example_activations = np.reshape(activations, (number_examples, num_layers, num_heads, head_dim))
        example_labels =[1 if label==True else 0 for label in df[df[id_column] == data_id]['correct'].values]
        
        separated_activations.append(example_activations)
        separated_labels.append(example_labels)
        
        data_ids_order.append(data_id)

    train_set_idxs, val_set_idxs, test_idxs = get_fold_indices(fold_index, args, data_ids_order)
    com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_activations, separated_labels)
    return com_directions, separated_activations

def run_sweep(tokenizer, model, sweep_eval_df, layers_to_sweep, heads_to_sweep_map, log, args):
    
    print("Starting attention head sweep...")
    if log == None: 
        sweep_results_summary = []
    else: 
        sweep_results_summary = log
    best_result = {"accuracy": -1.0, "layer": -1, "head": -1, "details": None}
    original_alpha = args.alpha
    print(sweep_eval_df.correct.values)

    for layer_idx in layers_to_sweep:
        heads_for_this_layer = heads_to_sweep_map.get(layer_idx, range(model.config.num_attention_heads))
        for head_idx in tqdm(heads_for_this_layer, desc=f"Sweeping Layer {layer_idx}"):
            
            heads = [(layer_idx, head_idx)]
            alphas = [args.alpha]
           
            results_raw = evaluate_configuration_general(args, tokenizer, model, heads, alphas, sweep_eval_df, external_test_set=sweep_eval_df.iloc[:])

            if args.prompt_type == "ab_cot":

                results = extract_answer_compare_gt(args, results_raw)

                curr_fold_results = pd.DataFrame(results)

                accuracy = curr_fold_results.correct.value_counts().to_dict().get(True, 0) / curr_fold_results.shape[0]

                metrics = {"accuracy": accuracy}
                    
                # Log and memoize the configuration
                log_entry = {
                    "heads": heads,
                    "alphas": [float(alpha) for alpha in alphas],
                    "seeds": args.seed,
                    "metrics": metrics,
                    "total": len(results_raw),
                    "consistency_factor": args.consistency_factor,
                }
                
                #log.append(log_entry)
                #log_filename = os.path.join(args.output_dir, "sweep_summary.csv")
                log_filename = "sweep_configuration.csv"

                args.output_path = args.output_dir + "/sweep_results_files"

                if not os.path.exists(args.output_path):
                    os.makedirs(args.output_path)

                append_to_log_file(args, log_filename, log_entry)
                fold_index = 0 
                #/results_files
                #args.output_path = os.path.join(args.output_dir,")
                
                save_results(args, heads , alphas, fold_index,  curr_fold_results, metrics)

                # Save the processed results to a file          

            correct_count = curr_fold_results.correct.value_counts().to_dict().get(True, 0)

            summary = {"heads":[(layer_idx, head_idx)], "accuracy": accuracy, "correct_count": correct_count, "total": len(results_raw), "metrics": {'accuracy': accuracy}}
            sweep_results_summary.append(summary)
            print(f"  L{layer_idx}H{head_idx}: Acc={accuracy:.4f} ({correct_count}/{len(results_raw)})")

            if accuracy > best_result["accuracy"]:
                best_result["accuracy"] = accuracy
                best_result["layer"] = layer_idx
                best_result["head"] = head_idx
                best_result["details"] = results

    args.alpha = original_alpha # Reset alpha
    print("Sweep finished.")
    print(f"Best performing head from sweep: Layer {best_result['layer']}, Head {best_result['head']} with Accuracy {best_result['accuracy']:.4f}")

    return best_result, sweep_results_summary

def run_sweep_layer(tokenizer, model, sweep_eval_df, layers_to_sweep, heads_to_sweep_map, log, external_data_set, args):
    
    print("Starting attention head sweep...")

    if log == None: 
        sweep_results_summary = []

    else: 
        sweep_results_summary = log

    best_result = {"accuracy": -1.0, "layer": -1, "head": -1, "details": None}

    original_alpha = args.alpha

    print(sweep_eval_df.correct.values)

    log_df = pd.DataFrame(sweep_results_summary)

    for layer_idx in layers_to_sweep:
        
        heads = [(layer_idx, head_idx) for head_idx in range(model.config.num_attention_heads)]
        alphas = [args.alpha for i in range(model.config.num_attention_heads)]
        
        
        if len(log_df) > 0:

            print(len(heads))

            if log_df['heads'].apply(lambda x: x == heads).any():
                print("Match found")
                
                #if log_df[(log_df['heads'] == heads) & (log_df['alphas'] == alphas)].shape[0] > 0:
                match = log_df[
                log_df['heads'].apply(lambda x: x == heads) & 
                log_df['alphas'].apply(lambda x: x == alphas)
                ]

                if not match.empty:
                
                    continue

        results_raw = evaluate_configuration_general(args, tokenizer, model, heads, alphas, sweep_eval_df, external_test_set=external_data_set)

        if args.prompt_type == "ab_cot":

            results = extract_answer_compare_gt(args, results_raw)

            curr_fold_results = pd.DataFrame(results)

            accuracy = curr_fold_results.correct.value_counts().to_dict().get(True, 0) / curr_fold_results.shape[0]

            metrics = {"accuracy": accuracy}
                
            # Log and memoize the configuration
            log_entry = {
                "heads": heads,
                "alphas": [float(alpha) for alpha in alphas],
                "seeds": args.seed,
                "metrics": metrics,
                "total": len(results_raw),
                "consistency_factor": args.consistency_factor,
            }
            
            #log.append(log_entry)
            #log_filename = os.path.join(args.output_dir, "sweep_summary.csv")
            log_filename = "sweep_configuration.csv"

            args.output_path = args.output_dir + "/sweep_results_files"

            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)

            append_to_log_file(args, log_filename, log_entry)
            fold_index = 0 
            #/results_files
            #args.output_path = os.path.join(args.output_dir,")
            
            save_results(args, heads , alphas, fold_index,  curr_fold_results, metrics)

            # Save the processed results to a file          

        correct_count = curr_fold_results.correct.value_counts().to_dict().get(True, 0)

        summary = {"heads":heads, "accuracy": accuracy, "correct_count": correct_count, "total": len(results_raw), "metrics": {'accuracy': accuracy}}
        sweep_results_summary.append(summary)
        #print(f"  L{layer_idx}H{head_idx}: Acc={accuracy:.4f} ({correct_count}/{len(results_raw)})")

        if accuracy > best_result["accuracy"]:
            best_result["accuracy"] = accuracy
            best_result["layer"] = layer_idx
            best_result["head"] = heads
            best_result["details"] = results

    args.alpha = original_alpha # Reset alpha
    print("Sweep finished.")
    print(f"Best performing head from sweep: Layer {best_result['layer']}, Head {best_result['head']} with Accuracy {best_result['accuracy']:.4f}")

    return best_result, sweep_results_summary


def prepare_evaluation_dataset(dataset_path, args):
    print(f"Loading evaluation dataset from: {dataset_path}")
    eval_df = pd.read_json(dataset_path)
    eval_df.reset_index(drop=True, inplace=True)
    
    # Apply consistency factor if needed, similar to prepare_test_set in notebook
    if args.consistency_factor > 1 and 'data_id' in eval_df.columns:
        unique_data_ids = eval_df.data_id.unique()
        if len(unique_data_ids) > 0:
            indexes = [eval_df[eval_df['data_id'] == data_id].index[0] for data_id in unique_data_ids]
            repeated_indexes = indexes * args.consistency_factor
            eval_df = eval_df.loc[repeated_indexes].reset_index(drop=True)

    print(f"Evaluation dataset prepared. Shape: {eval_df.shape}")
    return eval_df

def run_final_evaluation(tokenizer, model, sweep_summary,sweep_df, eval_df, args):

    best_result = {"accuracy": -1.0, "layer": -1, "head": -1, "details": None}

    configurations = pd.DataFrame(sweep_summary) 

    best_head_params = configurations[configurations['accuracy'] > 0.33]

    if len(best_head_params) > 5:

        max_accuracy = configurations['accuracy'].max()
        best_head_params = configurations[configurations['accuracy'] == max_accuracy]
    
    elif len(best_head_params) == 0:

        best_head_params = configurations[configurations['accuracy'] > 0]

    for heads in best_head_params['heads'].values:

        heads = heads

        alphas = [args.alpha]

        results_raw = evaluate_configuration_general(args, tokenizer, model, heads, alphas, sweep_df, external_test_set=eval_df)

        if args.prompt_type == "ab_cot":

            results = extract_answer_compare_gt(args, results_raw)

            curr_fold_results = pd.DataFrame(results)

            accuracy = curr_fold_results.correct.value_counts().to_dict().get(True, 0) / curr_fold_results.shape[0]

            metrics = {"accuracy": accuracy}
                
            # Log and memoize the configuration
            log_entry = {
                "heads": heads,
                "alphas": [float(alpha) for alpha in alphas],
                "seeds": args.seed,
                "metrics": metrics,
                "total": len(results_raw),
                "consistency_factor": args.consistency_factor,
            }
            
            #log.append(log_entry)
            #log_filename = os.path.join(args.output_dir, "sweep_summary.csv")
            
            args.output_path = args.output_dir + "/validation_results_files"
            
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)

            log_filename = "sweep_configuration.csv"

            append_to_log_file(args, log_filename, log_entry)
            fold_index = 0 

            save_results(args, heads , alphas, fold_index,  curr_fold_results, metrics)

            print(f"Results Layer {best_result['layer']} Head {best_result['head']} Evaluation Accuracy: {best_result['accuracy']:.4f}")

        if accuracy > best_result["accuracy"]:
            best_result["accuracy"] = accuracy
            best_result["layer"] = heads[0][0]
            best_result["head"] = heads[0][1]
            best_result["details"] = results
        
    ### change so it appends to final summary and also 
    final_summary = {
        "best_layer": best_result['layer'],
        "best_head": best_result['head'],
        "accuracy_on_eval_set": accuracy,
        "total": len(results_raw),
        "results": results
    }

    print(f"Final Best Results Layer {best_result['layer']} Head {best_result['head']} Evaluation Accuracy: {best_result['accuracy']:.4f}")
    
    return final_summary


import pandas as pd
from collections import OrderedDict

def keep_unique_heads_with_best_accuracy(ranked: pd.DataFrame) -> pd.DataFrame:
    """
    ranked : DataFrame already sorted by accuracy ↓
             • ranked["heads"] is a list of (layer, head) tuples

    Returns a DataFrame that keeps only those rows that
    introduce at least one head that has not appeared in a
    higher-accuracy row.
    """
    seen_heads = set()     # all heads we have kept so far
    keep_rows  = []        # the index numbers we want to keep

    for idx, row in ranked.iterrows():
        # Does this row have any *new* head?
        if any(h not in seen_heads for h in row["heads"]):
            keep_rows.append(idx)
            seen_heads.update(row["heads"])   # mark them as seen

    return ranked.loc[keep_rows]


def run_topk_combined_evaluation(
        tokenizer,
        model,
        sweep_summary,
        sweep_df,
        eval_df,
        args,
        k: int = 3,            # ← how many configs to merge
):
    """
    Combine the best `k` head-configurations into one larger set and evaluate it.

    Parameters
    ----------
    tokenizer, model, sweep_summary, sweep_df, eval_df, args : same as before
    k : int, optional
        Number of top configurations to combine (default: 3).

    Returns
    -------
    dict
        A summary with the combined heads, accuracy and raw results.
    """

    # # 1. Rank configurations
    # configs = pd.DataFrame(sweep_summary)
    # ranked = configs.sort_values("accuracy", ascending=False)

    # if ranked.empty:
    #     raise ValueError("No configurations with accuracy > 0 found.")

    # # 2. Pick the top-k and gather their heads
    # top_k = ranked.head(k)["heads"].values

 
    # # 3) Merge heads and alphas, deduping heads but
    # # keeping the alpha that appears first
    # combined = OrderedDict()          # (layer, head) → alpha
    # for _, row in top_k.iterrows():
    #     heads  = row["heads"]         # e.g. [(9,3), (10,5)]
    #     alphas = row["alphas"]        # e.g. [0.8, 0.6]
    #     if len(heads) != len(alphas):
    #         raise ValueError("Heads and alphas length mismatch.")
        
    #     for h, a in zip(heads, alphas):
    #         if h not in combined:     # keep only first alpha for duplicates
    #             combined[h] = a

    # 1) Make a DataFrame and sort by accuracy (high → low)
    ranked = pd.DataFrame(sweep_summary).sort_values("accuracy", ascending=False)

    # 2) Drop rows that would only repeat heads we already saw
    ranked = keep_unique_heads_with_best_accuracy(ranked)

    if ranked.empty:
        raise ValueError("No configurations with accuracy > 0.")

    # 2) Pick the best k *unique* heads and their matching alphas
    from collections import OrderedDict
    combined = OrderedDict()           # (layer, head) → alpha
    
    top_ks = [1,2,3,4]
    
    for k in top_ks:

        i=0
        for _, row in ranked.iterrows():
            
            i+=1
            heads  = row["heads"]          # e.g. [(9, 3), (10, 5)]
            print(heads)
            alphas = row["alphas"]         # e.g. [0.7, 0.6]
            if len(heads) != len(alphas):
                raise ValueError("Heads and alphas length differ.")

            for h, a in zip(heads, alphas):
                if h not in combined:      # keep only the first (best) α per head
                    combined[h] = a
                    #if len(combined) == k: # stop after k distinct heads
                    #    break
            #if len(combined) == k:
            #    break
            if i == k: 
                break

        combined_heads  = list(combined.keys())
        combined_alphas = list(combined.values())
        

        alpha_range = [0.5, 0.6, 0.7, 0.8, 0.9,1.0]  # [0.5, 0.6
        #alpha_scaling = 0.5 # scaling factor for the combined alpha
        
        for alpha_scaling in alpha_range:
            print(f"Top {k} unique heads: {combined_heads}, with alphas scaled by {alpha_scaling}")
            # 4. Run the evaluation
            alphas = [alpha*alpha_scaling for alpha in combined_alphas]

            results_raw = evaluate_configuration_general(
                args, tokenizer, model,
                combined_heads, alphas,
                sweep_df,
                external_test_set=eval_df
            )

            # 5. Compute simple accuracy if prompt_type == "ab_cot"
            if args.prompt_type == "ab_cot":
                processed = extract_answer_compare_gt(args, results_raw)
                df = pd.DataFrame(processed)
                accuracy = df.correct.mean()            # mean() is same as (#True / n)

                fold_index = 0 

                metrics = {"accuracy": accuracy}

                args.output_path = args.output_dir + "/validation_results_files"
                    
                if not os.path.exists(args.output_path):
                    os.makedirs(args.output_path)

                log_filename = "sweep_configuration.csv"

                log_entry = {
                "heads": combined_heads,
                "alphas": [float(alpha) for alpha in alphas],
                "seeds": args.seed,
                "metrics": metrics,
                "total": len(results_raw),
                "consistency_factor": args.consistency_factor,
                    }

                
                append_to_log_file(args, log_filename, log_entry)


                #save_results(args, heads , alphas, fold_index,  processed, metrics)

                print(f"Combined Configuration Evaluation Accuracy: {accuracy:.4f}")





            else:
                accuracy = None                         # or add an else-branch as needed
            

            summary = {
                "heads": combined_heads,
                "accuracy": accuracy,
                "total_eval_examples": len(results_raw),
                "raw_results": results_raw,
            }

    # Optional: save / log just like in run_final_evaluation
    return summary

def main():
    args = parse_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = select_device(min_vram_gb=20)
    
    # 1. Load Model and Tokenizer
    print(f"Loading model: {args.model_name}")
    device = "auto"
    tokenizer, model = load_model(args.model_name, device)

    #model.to(device)#(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    try: 
        args.num_heads = model.config.num_attention_heads # Set num_heads in args
    
    except AttributeError:
        args.num_heads = model.config.text_config.num_attention_heads  # Set num_heads in args
    
    #args.num_fold = 1
    #args.val_ratio = 0
    args.suffix = ""
    # 2. Load Initial Data and Calculate Directions
    initial_df = load_and_preprocess_initial_data(args.initial_dataset_path, tokenizer, model, args)

    #args.output_path = args.output_dir
    if args.alpha == "none": 
        alphas = json.loads(args.alphas)
    else: 
        alphas = [float(args.alpha)]

    sweep = True#False #True #False # True #False #True #False #True #False #True

    sweep_heads = False
    sweep_layers = True
    recombination = False

    log_filename = "sweep_configuration.csv"

    log_filename_complete = f"{args.output_dir}/sweep_results_files/{log_filename}"
    print(f"Loading memoization and log from {log_filename}")

    # Initialize variables
    memoization = []
    configurations = None
    if os.path.exists(log_filename_complete):
        try:

            log_df = pd.read_csv(log_filename_complete, converters={
            'heads': ast.literal_eval,
            'alphas': ast.literal_eval,
            'seeds': ast.literal_eval,
            'metrics': ast.literal_eval,
            })

            #print(log_df.metrics.values[0])
            ## --> GET UNIQUE HEAD COMBINATIONS
            #unique_head_combinations = set(log_df["heads"])#.apply(eval).apply(tuple))
            #print(unique_head_combinations)
            ## log_df to log
            log = log_df.to_dict('records')

            for configuration in log:

                #heads_frozenset = frozenset([tuple(h) for h in heads])
                heads = configuration['heads']
                # configuration['consistency_factor']
                heads_tuple = (tuple(sorted([tuple(head) for head in heads])), configuration['alphas'])
                print(heads_tuple)
                memoization.append(heads_tuple)

            print(f"Loaded {len(log)} configurations from the log file.")
            print(memoization)
            configurations = log_df.to_dict('records')
            print(configurations[0])

        except pd.errors.EmptyDataError:
            print(f"File '{log_filename_complete}' is either empty or has no columns to parse.")
            # Debugging: show the raw content of the file
            with open(log_filename_complete, 'r') as file:
                content = file.read().strip()
                if not content:
                    print(f"'{log_filename_complete}' is completely empty.")
                else:
                    print(f"Content of the file:\n{content}")

    else:
        print("No existing log file found. Starting fresh.")

    # 5. Load Final Evaluation Dataset
    final_eval_df = prepare_evaluation_dataset(args.evaluation_dataset_path, args)
    final_eval_df = final_eval_df.iloc[0:100]

    for alpha in alphas:

        args.alpha = alpha

        if sweep:
            # 3. Parse layers and heads for sweep
            try:
                layers_to_sweep = [int(l.strip()) for l in args.layers_to_sweep.split(',')]
            except ValueError:
                print("Error: layers_to_sweep must be comma-separated integers (e.g., '10,11,12').")
                return

            heads_to_sweep_map = {} # layer_idx -> list of head_indices
            
            # 4. Run Sweep
            if args.heads_to_sweep_per_layer:
                try:
                    head_indices = [int(h.strip()) for h in args.heads_to_sweep_per_layer.split(',')]
                    for l_idx in layers_to_sweep:
                        heads_to_sweep_map[l_idx] = head_indices
                except ValueError:
                    print("Error: heads_to_sweep_per_layer must be comma-separated integers (e.g., '0,1,5').")
                    return
            else: # If not specified, sweep all heads for the layers_to_sweep
                
                for l_idx in layers_to_sweep:
                    heads_to_sweep_map[l_idx] = []
                    
                    #heads_to_sweep_map[l_idx] = list(range(model.config.num_attention_heads))
                    
                    for h_idx in range(args.num_heads):

                        if (((l_idx, h_idx),), [args.alpha]) not in memoization:

                            heads_to_sweep_map[l_idx].append(h_idx)
        
        else:
            layers_to_sweep = []
            heads_to_sweep_map = {}


        if sweep_heads:
            
            best_head_sweep_result, sweep_summary_list = run_sweep(tokenizer, model, initial_df, layers_to_sweep, heads_to_sweep_map, configurations, args) # ,
            for i in sweep_summary_list:
                #print(i)
                i['accuracy'] = i['metrics']['accuracy'] #eval(
        
        elif sweep_layers:
            best_layer_sweep_result, sweep_summary_list = run_sweep_layer(tokenizer, model, initial_df, layers_to_sweep, heads_to_sweep_map, configurations, final_eval_df, args) #,
            for i in sweep_summary_list:
                #print(i)
                i['accuracy'] = i['metrics']['accuracy'] #eval(
        # Save sweep summary
        #sweep_summary_df = pd.DataFrame(sweep_summary_list)
        #sweep_summary_path = os.path.join(args.output_dir, "sweep_summary.csv")
        #sweep_summary_df.to_csv(sweep_summary_path, index=False)
        #print(f"Sweep summary saved to {sweep_summary_path}")
        
        # if best_head_sweep_result["layer"] == -1:
        #     print("No successful sweep iteration. Cannot proceed to final evaluation.")
        #     return
        

    
    if sweep_heads:
        # 6. Run Final Evaluation
        final_evaluation_summary = run_final_evaluation(tokenizer, model, sweep_summary_list, initial_df, final_eval_df, args)#args, args)

        # # Save final evaluation results
        # final_eval_path = os.path.join(args.output_dir, "final_evaluation_results.json")
        # with open(final_eval_path, 'w') as f:
        #     json.dump(final_evaluation_summary, f, indent=4, cls=NpEncoder)
        # print(f"Validation results safed to {final_eval_path}")

    if recombination: 


        sweep_summary_list = configurations

        for i in sweep_summary_list:
            #print(i)
            i['accuracy'] = i['metrics']['accuracy'] #eval(

        recombination_results = run_topk_combined_evaluation(tokenizer, model, sweep_summary_list, initial_df, final_eval_df, args) 

    print("Script finished.")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    main()