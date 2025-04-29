import argparse
import heapq
import pandas as pd
import os
import torch
import numpy as np

import json
import time 
from ut_evaluation_utils import evaluate_configuration, append_to_log_file
## optimize_alpha_for_heads,
from ut_processing_utils import (load_model,
                                select_device, 
                                prepare_test_set,
                                process_data,
                                get_fold_indices,
                                get_best_head_metrics,
                                get_best_head_metrics_seeds,      
                                ParseListOfLists
)

import ast

from ut_intervention_utils import (run_llama_intervention_batch_parallel,
                                   get_com_directions,
                                   get_interventions_dict_variable_alpha
)

def optimize_alpha_for_heads(
    args,
    tokenizer,
    model,
    heads, 
    alphas,
    num_heads,
    memoization,
    log,
    log_filename,
    optimized_alphas,
    ):
    
    best_alphas = None
    best_precision = 0
    best_recall = 0
    best_metrics = {"precision": 0, "recall": 0}  
    alpha_step = 0.5 #0.1  # Initialize as a relative step size (10%)
    direction = 1  # 1 for increasing, -1 for decreasing
    max_iterations = 6  # Set a reasonable upper limit to prevent infinite loops
    no_improve_counter = 0  # Counter for iterations without improvement
    required_no_improve = 3  # Number of consecutive non-improving iterations to trigger stopping

    iteration = 0
    
    

    while iteration < max_iterations and no_improve_counter < required_no_improve:
        
        iteration += 1
        print(f"--- Iteration {iteration} ---")
        print("Current heads:", heads)
        print("Current alphas:", alphas)
        
        # config_key = tuple(sorted(zip(heads, alphas)))
        
        # if config_key in memoization:
        #     print("Using memoized results for configuration:", config_key)
        #     # Skip the rest of the loop and continue to the next iteration
        #     no_improve_counter += 1
        #     if no_improve_counter >= required_no_improve:
        #         print("No improvement after using memoized results. Stopping optimization.")
        #         break
        #     continue
        
        precision_scores, recall_scores, undefined = evaluate_configuration(
            args=args,
            tokenizer=tokenizer,
            model=model,
            heads=heads,
            alphas=alphas,
            num_heads=num_heads,
        )
        
        #time.sleep(60)  # Wait for 60 seconds to avoid overwhelming the server

        precision = round(np.mean([entry["precision"] for entry in precision_scores]), 2)
        recall = round(np.mean([entry["recall"] for entry in recall_scores]), 2)

        # Update best configuration if performance improves
        if precision > best_precision or (precision == best_precision and recall > best_recall):
            print("New best configuration found!")
            best_alphas = alphas#.copy()
            best_heads = heads #.copy()
            best_precision = precision
            best_recall = recall
            best_metrics = {"precision": precision, "recall": recall}
            print("Heads for logging: ", heads)
            print("Best alphas:", best_alphas)
            print("Best metrics:", best_metrics)
            no_improve_counter = 0  # Reset counter since improvement was found
        else:
            no_improve_counter += 1
            print(f"No improvement in this iteration. No improvement counter: {no_improve_counter}")

        # Log and memoize the configuration
        log_entry = {
            "heads": heads,
            "alphas": [float(alpha) for alpha in alphas],
            "precision": precision,
            "recall": recall,
            "seeds": args.seeds,
            
        }

        log.append(log_entry)
        append_to_log_file(args, log_filename, log_entry)
        
        #memoization.add(config_key)  # Assuming memoization is a set. If it's a dict, adjust accordingly.

        # Adjust alphas based on precision
        if precision == 1 or undefined == 1: 
            if direction == 1:
                alpha_step *= 0.4  # Reduce step size when changing direction
            else: 
                alpha_step *= 1.15
            direction = -1
            alphas = [alpha * (1 - alpha_step) for alpha in alphas]
            print("Decreasing alphas:", alphas)
        
        elif precision < 1:
            if direction == -1:
                alpha_step *= 0.4  # Reduce step size when changing direction
            else: 
                alpha_step *= 1.15
            direction = 1
            alphas = [alpha * (1 + alpha_step) for alpha in alphas]
            print("Increasing alphas:", alphas)
            
        else:
            print("Problem: Unexpected precision value!")
            break

        # Ensure alphas stay within a reasonable range
        alphas = [max(0, min(alpha, 600)) for alpha in alphas]

        # Stop if the step size becomes too small
        if alpha_step < 0.1*alpha_step / len(heads):
            print("Alpha step size too small. Stopping optimization.")
            break

        print(iteration < max_iterations and no_improve_counter < required_no_improve)


    # After optimization, store the best alphas found
    if best_alphas != None:
        optimized_alphas[tuple(best_heads)] = {
            "alphas": best_alphas,
            "precision": best_precision,
            "recall": best_recall
        }

    return best_alphas, best_metrics


def split_and_optimize_heads_iterative(
    configurations,
    alpha_step,
    args,
    tokenizer,
    model,
    num_heads,
    memoization,
    log,
    log_filename_input,
    log_filename,
    min_precision=0.0,
    min_recall=0.0,
    max_iterations=1000,
):
    # Initialize the priority queue
    queue = []
    iterations = 0
    overall_best_config = None
    optimized_alphas = {}  # Store optimized alphas for each head

    # Initialize variables
    memoization = {}
    
    log_filename_complete = f"{args.output_path}/{log_filename_input}"
    print(f"Loading memoization and log from {log_filename_input}")

    if os.path.exists(log_filename_complete):

        try:

            log_df = pd.read_csv(log_filename_complete, converters={
            'heads': ast.literal_eval,
            'alphas': ast.literal_eval,
            'seeds': ast.literal_eval
            })

            ## --> GET UNIQUE HEAD COMBINATIONS
            unique_head_combinations = set(log_df["heads"])#.apply(eval).apply(tuple))
            print(unique_head_combinations)
            ## log_df to log
            log = log_df.to_dict('records')

            for heads in unique_head_combinations:

                #heads_frozenset = frozenset([tuple(h) for h in heads])
                heads_frozenset = heads
                ## now add to memoization heads tried with best alphas so far
                if heads_frozenset not in memoization:
                    memoization[heads_frozenset] = {
                        'seeds': set(),
                        'alphas': None,
                        'precision': None,
                        'recall':None
                    }

                #print(heads_frozenset)
                best_config = get_best_head_metrics_seeds(log, heads_frozenset)

                for seed in best_config["seeds"]:
                    memoization[heads_frozenset]['seeds'].add(seed)

                memoization[heads_frozenset]['alphas'] = best_config["alphas"]  # Assume the best alphas per head combination
                memoization[heads_frozenset]['precision'] = best_config["precision"]
                memoization[heads_frozenset]['recall'] = best_config["recall"]


                optimized_alphas[heads_frozenset] = {'alphas': best_config["alphas"], 
                                                     'precision': best_config["precision"],
                                                     'recall': best_config["recall"]}
                

            print(f"Loaded {len(log)} configurations from the log file.")
            

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

    # Initialize best solution
    overall_best_config = {"heads": [], "alphas": [], "precision": 0, "recall": 0}
    # Update best solution from loaded log if available
    
    if log:

        best_entry = max(log, key=lambda x: (x["precision"], x["recall"]))
        
        overall_best_config = {
            "heads": best_entry["heads"],
            "alphas": best_entry["alphas"],
            "precision": best_entry["precision"],
            "recall": best_entry["recall"],
        }

        min_recall = best_entry["recall"]*0.5 


    for count, key in enumerate(list(memoization.keys())): 

        if not all(seed in memoization[key]['seeds'] for seed in args.seeds):
            print(key, "does not contain all seeds.")
            
            best_config = memoization[key]#['results']
        
            #print(best_config)
            #print(best_config["precision"])
            #print(best_config["recall"])

            if best_config['precision'] >= min_precision and best_config['recall'] >= min_recall:

                config = {
                    "heads": key,
                    "alphas": memoization[key]["alphas"],
                    "precision": memoization[key]["precision"],
                    "recall": memoization[key]["recall"],
                }

                # Push both halves to the heap with their priority
                heapq.heappush(queue, (
                    -memoization[key]["precision"],  # Negative because heapq is a min-heap
                    -memoization[key]["recall"],
                    len(key),  # For tie-breaking
                    key,
                    count,
                    config))
            
            del memoization[key]
        
    print("Queue: ", queue)
    
    while queue and iterations < max_iterations:

        print("Length of the queue: ", len(queue))
        count+=1
        iterations += 1
        # Pop the subset with the highest priority (highest precision and recall)
        _, _, _, _,_, config = heapq.heappop(queue)

        current_heads = config["heads"]
        current_alphas = config["alphas"]

        heads_tuple = tuple(sorted([tuple(head) for head in current_heads]))
        
        if heads_tuple in memoization.keys():
            #print("Skipping configuration: Already evaluated.")
            #print(heads_tuple)
            continue

        print(heads_tuple)
        print(len(heads_tuple))
        
        memoization[heads_tuple] = {
                    'seeds': set(),
                    'alphas': None,
                    'precision': None,
                    'recall':None
                    }
        
        for seed in args.seeds:
            memoization[heads_tuple]['seeds'].add(seed)
        #tried_head_combinations.add(frozenset(heads_tuple))

        best_alphas, metrics = optimize_alpha_for_heads(
            args=args,
            tokenizer=tokenizer,
            model=model,
            heads=heads_tuple,
            alphas=current_alphas,
            num_heads=num_heads,
            memoization=memoization,
            log=log,
            log_filename=log_filename,
            optimized_alphas=optimized_alphas,
        )

        memoization[heads_tuple]['alphas'] = best_alphas
        memoization[heads_tuple]['precision'] = metrics['precision']
        memoization[heads_tuple]['recall'] = metrics['recall']

        # Store the result
        optimized_alphas[heads_tuple] = {
            'heads': heads_tuple,
            'alphas': best_alphas,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
        }

        print(queue)

    return overall_best_config, log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama_7B", help="model name")
    #parser.add_argument("--num_heads", type=int, default=48, help="K, number of top heads to intervene on")
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds")
    parser.add_argument(
        "--val_ratio", type=float, help="ratio of validation set size to development set size", default=0.5
    )
    parser.add_argument(
        "--use_center_of_mass",
        type=lambda x: str(x).lower() == "true",
        help="Whether to use the center of mass or not",
    )
    parser.add_argument(
        "--use_random_dir", action="store_true", help="use random direction", default=False
    )
    #parser.add_argument("--seed", type=int, default=42, help="seed")
    #parser.add_argument("--layer", type=int, help="layer for intervention")
    # Define argument parser
    parser.add_argument("--seeds", type=str, default="[1234,5678,9012]", help="seeds as a JSON array")
    parser.add_argument("--consistency_factor", type=int, default=6)
    parser.add_argument("--input_path", type=str, help="input path")
    parser.add_argument("--output_path", type=str, help="output path")
    parser.add_argument("--dataset_name", type=str, default="requirements_data")
    parser.add_argument(
        "--add_or_subtract",
        type=lambda x: str(x).lower() == "true",
        default="true",
        help="if intervention is added or substract to activations",
    )

    #parser.add_argument("--list_of_heads", type=str, default="", action=ParseListOfLists, help="Input should be a list of lists (e.g., [['11', '0'], ['13', '0']]).")
    parser.add_argument("--prompt_type", type=str, default="ab_cot")
    parser.add_argument("--temperature", type=float, default=0.8)
    #parser.add_argument("--consistency_factor", type=int, default=6)

    args = parser.parse_args()

    # Convert the JSON string to a Python list
    args.seeds = json.loads(args.seeds)

    if not os.path.exists(f"{args.output_path}"):
        os.mkdir(f"{args.output_path}")

    # Print all arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Initialize variables
    memoization = set()
    log = []
    #initial_alpha = 20#args.alpha  # Starting alpha value
    alpha_step = 5
    min_precision = 1.0  # Minimum precision for a head to be considered effective
    min_recall = .0  # Minimum recall for a head to be considered effective
    
    num_heads = 32# args.num_heads  # From args

    #args.seeds =[1234, 5678, 9012]

    args.add_or_subtract = False

    # Print all arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    tokenizer, model = load_model(args.model_name)

    log = []
    log_filename_input = "configuration_optimization.csv"
    log_filename = "configuration_optimization_multi_fidelity.csv"
    log_path = f"{args.output_path}/{log_filename}"
    
    #layers = [12,13,15]
    #alphas = [1.21875, 2.625, 4.625]
    #alphas = [3, 3, 3]
    
    
    layers = [i for i in range(32)]
    alphas = [4]*len(layers)
    
    configurations = []

    for layer, alpha in zip(layers, alphas):

        configurations.append(([[layer, head] for head in range(0,32)], alpha))

    best_configuration, log = split_and_optimize_heads_iterative(
        configurations=configurations,
        alpha_step=alpha_step,
        args=args,
        tokenizer=tokenizer,
        model=model,
        num_heads=num_heads,
        memoization=memoization,
        log=log,
        log_filename_input=log_filename_input,
        log_filename=log_filename,
        min_precision=min_precision,
        min_recall=min_recall,
    )

    print("Best Configuration:")
    print("Heads:", best_configuration["heads"])
    print("Alphas:", best_configuration["alphas"])
    print("Precision:", best_configuration["precision"])
    print("Recall:", best_configuration["recall"])

if __name__ == "__main__":
    main()
