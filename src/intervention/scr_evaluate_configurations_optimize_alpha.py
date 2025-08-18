import argparse
import heapq
import pandas as pd
import os
import torch
import numpy as np

import json
import time 
from utils.ut_evaluation_utils import evaluate_configurationappend_to_log_file, optimize_alpha_for_heads

from utils.ut_processing_utils import (load_model,
                                select_device, 
                                prepare_test_set,
                                process_data,
                                get_fold_indices,
                                get_best_head_metrics,
                                get_best_head_metrics_seeds,      
                                ParseListOfLists
)

import ast

from utils.ut_intervention_utils import (run_llama_intervention_batch_parallel,
                                   get_com_directions,
                                   get_interventions_dict_variable_alpha
)

def optimize_alpha_for_heads(
    args,
    tokenizer,
    model,
    heads, 
    alphas,
    log,
    log_filename,
    optimized_alphas,
    ):
    
    best_alphas = None
    best_precision = 0
    best_recall = 0
    best_metrics = {"precision": 0, "recall": 0}  
    alpha_step = 0.1 #0.1  # Initialize as a relative step size (10%)
    direction = 1  # 1 for increasing, -1 for decreasing
    max_iterations = 1  # Set a reasonable upper limit to prevent infinite loops
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
            "consistency_factor": args.consistency_factor,
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



def run_configurations(
    configurations,
    args,
    tokenizer,
    model,
    log,
    log_filename,
):
    
    overall_best_config = None
    optimized_alphas = {}  # Store optimized alphas for each head

    memory = False

    # Initialize variables
    memoization = []
    
    log_filename_complete = f"{args.output_path}/{log_filename}"
    print(f"Loading memoization and log from {log_filename}")

    if os.path.exists(log_filename_complete):

        try:

            log_df = pd.read_csv(log_filename_complete, converters={
            'heads': ast.literal_eval,
            'alphas': ast.literal_eval,
            'seeds': ast.literal_eval
            })

            ## --> GET UNIQUE HEAD COMBINATIONS
            #unique_head_combinations = set(log_df["heads"])#.apply(eval).apply(tuple))
            #print(unique_head_combinations)
            ## log_df to log
            log = log_df.to_dict('records')

            for configuration in log:

                #heads_frozenset = frozenset([tuple(h) for h in heads])
                heads = configuration['heads']

                heads_tuple = (tuple(sorted([tuple(head) for head in heads])), configuration['consistency_factor'])
                print(heads_tuple)
                memoization.append(heads_tuple)

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

        # min_recall = best_entry["recall"]*0.5 

    
        
        for configuration in configurations:

            alphas = configuration["alphas"]
            heads = configuration["heads"]

            enhanced_heads_tuple = (tuple(sorted([tuple(head) for head in heads])), args.consistency_factor)
            
            if memory: 
                if enhanced_heads_tuple in memoization:
                    print("Skipping configuration: Already evaluated.")
                    print(enhanced_heads_tuple)
                    continue

            heads_tuple = tuple(sorted([tuple(head) for head in heads]))
            print(heads_tuple)

            best_alphas, metrics = optimize_alpha_for_heads(
            args=args,
            tokenizer=tokenizer,
            model=model,
            heads=heads_tuple,
            alphas=alphas,
            log=log,
            log_filename=log_filename,
            optimized_alphas=optimized_alphas,
            )


            
    return overall_best_config, log

def main():

    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama_7B", help="model name")
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

    parser.add_argument("--prompt_type", type=str, default="ab_cot")

    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument("--consistency_factors", type=str, default="[5]")

    args = parser.parse_args()

    # Convert the JSON string to a Python list
    args.seeds = json.loads(args.seeds)

    # Convert the JSON string to a Python list
    args.consistency_factors = json.loads(args.consistency_factors)

    if not os.path.exists(f"{args.output_path}"):
        os.mkdir(f"{args.output_path}")

    log = []

    args.add_or_subtract = False

    # Print all arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    tokenizer, model = load_model(args.model_name)

    log = []
    log_filename_input = "configuration_optimization.csv"
    
    log_filename = "configuration_optimization_consistency_experiments.csv"
    log_path = f"{args.output_path}/{log_filename}"

    log_filename_complete = f"{args.output_path}/{log_filename_input}"
    
    configurations = pd.read_csv(log_filename_complete, converters={
            'heads': ast.literal_eval,
            'alphas': ast.literal_eval,
            'seeds': ast.literal_eval
            }) 

    #print(configurations)
    # #configurations = configurations[(configurations['precision'] == 1)]
    # unique_head_combinations = set(configurations["heads"])#.apply(eval).apply(tuple))
    
    # #print(unique_head_combinations)
    
    # log = configurations.to_dict('records')
    # configurations = []
    # for heads in unique_head_combinations:
    
    #     best_config = get_best_head_metrics_seeds(log, heads)
    #     best_config["heads"] = heads
    
    #     configurations.append(best_config)

    # configurations = pd.DataFrame(configurations)

    sorted_configurations = configurations[(configurations['precision'] > 0.99) 
                   #  & (configurations['seeds'].apply(lambda seed_list: 5678 in seed_list))
    ].sort_values(by='recall', ascending=False).iloc[0:1]
    
    print(sorted_configurations)
    
    # Define the percentage increments
    percentages = [i for i in range(0, 40, 10)]  # 5% to 100%

    # Function to expand the DataFrame
    def expand_alphas(df):
        rows = []
        for _, row in df.iterrows():
            alphas = row['alphas']
            for p in percentages:
                new_row = row.copy()
                factor = 1 + p / 100.0
                new_row['alphas'] = [alpha * factor for alpha in alphas]
                new_row['percentage_increase'] = p
                rows.append(new_row)
        return pd.DataFrame(rows)

    # Expand the DataFrame
    sorted_configurations = expand_alphas(sorted_configurations)
    sorted_configurations = sorted_configurations.to_dict('records')
    print(sorted_configurations)
    #sorted_configurations = configurations.to_dict('records')
   
    # configurations = configurations.drop_duplicates(subset=['heads'])

    # sorted_configurations = configurations[(configurations['precision'] == 1) 
    #                 # & (df['seeds'].apply(lambda seed_list: 5678 in seed_list))
    # ].sort_values(by='recall', ascending=False).head(5)

    #sorted_configurations.append({'heads': ((0, 0),), 'alphas': [0]})
    sorted_configurations = [{'heads': ((0, 0),), 'alphas': [0]}]
    
    # #sorted_configurations = [{'heads': ((15, 4), (15, 5), (15, 6), (15, 7)), 'alphas':[50.2793488293372, 50.2793488293372, 50.2793488293372, 50.2793488293372]}]

    print(sorted_configurations)
    num_mc = 1 
    for i in range(num_mc):

        best_configuration, log = run_configurations(
            configurations=sorted_configurations,
            args=args,
            tokenizer=tokenizer,
            model=model,
            log=log,
            #log_filename_input=log_filename_input,
            log_filename=log_filename,
        )

    print("Best Configuration:")
    print("Heads:", best_configuration["heads"])
    print("Alphas:", best_configuration["alphas"])
    print("Precision:", best_configuration["precision"])
    print("Recall:", best_configuration["recall"])

if __name__ == "__main__":
    main()
