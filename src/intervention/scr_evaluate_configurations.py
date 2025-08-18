import argparse
import heapq
import pandas as pd
import os
import torch
import numpy as np

import copy

import json
import time 
from utils.ut_evaluation_utils import evaluate_configuration, append_to_log_file
## optimize_alpha_for_heads,
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

                heads_tuple = (tuple(sorted([tuple(head) for head in configuration['heads']])),configuration['alphas'], configuration['consistency_factor'])
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
    print("Configurations: ", configurations)
    for configuration in configurations:

        alphas = configuration["alphas"]
        heads = configuration["heads"]

        #heads_frozenset = frozenset([tuple(h) for h in heads])
        heads = configuration['heads']
        if type(heads[0]) == int:
            print(heads)
            heads = (heads,)

        heads_tuple = (tuple(sorted([tuple(head) for head in heads])), alphas, args.consistency_factor)
        print(heads_tuple)

        if memory: 
            if heads_tuple in memoization:
                print("Skipping configuration: Already evaluated.")
                print(heads_tuple)
                continue
        
        #if args.test_set != None:
        if args.test_set and args.test_set.lower() != "null":
            test_set = args.test_set
            print("load test set..")
            test_set = pd.read_json(test_set)
        
        else:
            test_set = None

        print("Test set:" , test_set)
        precision_scores, recall_scores, undefined = evaluate_configuration(
        args=args,
        tokenizer=tokenizer,
        model=model,
        heads=heads,
        alphas=alphas,
        external_test_set=test_set,
        )

        precision = round(np.mean([entry["precision"] for entry in precision_scores]), 2)
        recall = round(np.mean([entry["recall"] for entry in recall_scores]), 2)
        precision_consistency = round(np.mean([entry["precision_consistency"] for entry in precision_scores]), 2)
        recall_consistency = round(np.mean([entry["recall_consistency"] for entry in recall_scores]), 2)

        # Log and memoize the configuration
        log_entry = {
            "heads": heads,
            "alphas": [float(alpha) for alpha in alphas],
            "precision": precision,
            "recall": recall,
            "seeds": args.seeds,
            "consistency_factor": args.consistency_factor,
            "precision_consistency": precision_consistency,
            "recall_consistency": recall_consistency,
            "temperature": args.temperature,
        }

        log.append(log_entry)
        append_to_log_file(args, log_filename, log_entry)

    return overall_best_config, log

def none_to_zero(value):
    """
    Converts None, 'None', 'null', 'NULL', '', or non-integer values to 0.

    Parameters:
    value (any): The value to be converted.

    Returns:
    int: The converted value, which is 0 if the input value is None, 'None', 'null', 'NULL', '', or not an integer.
    """
    if value in [None, 'None', 'null', 'NULL', '']:
        return 0
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")

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

    parser.add_argument("--input_path_configuration", type=str, help="path to configuration CSV file", default=None)

    parser.add_argument("--limit", type=int, default=1, help="limit the number of configurations to evaluate")
    parser.add_argument("--test_set", type=str, default=None, help="path to test set CSV file")
    parser.add_argument("--prompt_type", type=str, default="ab_cot")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--consistency_factors", type=str, default="[5]")
    parser.add_argument("--temperatures", type=str, default="[]")
    
    parser.add_argument("--num_mc", type=int, default=1)
    parser.add_argument("--alpha_sweep", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--num_sequences", type=int, default=1)

    parser.add_argument("--upper_bound", type=none_to_zero, default=30)
    parser.add_argument("--lower_bound", type=none_to_zero, default=0)

    args = parser.parse_args()

    args.temperatures = json.loads(args.temperatures)

    if args.temperatures == "[]" or args.temperatures == None:
        args.temperatures = [args.temperature]
    
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


    # 1. Load Model and Tokenizer
    print(f"Loading model: {args.model_name}")

    device = "auto"

    tokenizer, model = load_model(args.model_name, device)

    log = []

    print(args.input_path_configuration)

    # Check if input_path_configuration is not None or "null"
    if args.input_path_configuration and args.input_path_configuration.lower() != "null":
        log_filename_input = args.input_path_configuration
    else:
        log_filename_input = "configuration_optimization.csv"
    
    print(f"Log file: {log_filename_input}")

    ## join file path with os
    log_filename_complete = os.path.join(args.output_path, log_filename_input)
    print(f"Log file: {log_filename_complete}")
    #log_filename_complete = f"{args.output_path}/{log_filename_input}"

    #log_filename = "configuration_optimization_manual_output.csv"
    log_filename = "configuration_optimization_consistency_experiments.csv"
    log_path = f"{args.output_path}/{log_filename}"

    configurations = pd.read_csv(log_filename_complete, converters={
            'heads': ast.literal_eval,
            'alphas': ast.literal_eval,
            'seeds': ast.literal_eval
            }) 

    #configurations = configurations[(configurations['precision'] == 1)]
    unique_head_combinations = set(configurations["heads"])#.apply(eval).apply(tuple))
    print(unique_head_combinations)
    
    log = configurations.to_dict('records')
    configurations = []
    for heads in unique_head_combinations:
    
        best_config = get_best_head_metrics_seeds(log, heads)
        best_config["heads"] = heads
    
        configurations.append(best_config)

    configurations = pd.DataFrame(configurations)
    
    sorted_configurations = configurations.assign(seeds_length=configurations['seeds'].apply(len)).sort_values(
    by=['seeds_length','precision', 'recall'], 
    ascending=[False, False, False]
    )[0:args.limit]
    
    print("Configuration: ", sorted_configurations)
    if args.alpha_sweep:
        # Define the percentage increments
        percentages = [i for i in range(args.lower_bound, args.upper_bound, 10)]  # 5% to 100%

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

        sorted_configurations = expand_alphas(sorted_configurations)
    sorted_configurations = sorted_configurations.to_dict('records')

    #sorted_configurations = configurations.to_dict('records')
    # configurations = configurations.drop_duplicates(subset=['heads'])
    # sorted_configurations = configurations[(configurations['precision'] == 1) 
    #                 # & (df['seeds'].apply(lambda seed_list: 5678 in seed_list))
    # ].sort_values(by='recall', ascending=False).head(5)
    #sorted_configurations.append({'heads': ((0, 0),), 'alphas': [0]})
    #sorted_configurations = [{'heads': ((0, 0),), 'alphas': [0]}]
    #sorted_configurations = [{'heads': ((0, 0),), 'alphas': [0]}]
    #sorted_configurations = [{'heads': ((15, 4), (15, 5), (15, 6), (15, 7)), 'alphas':[50.2793488293372, 50.2793488293372, 50.2793488293372, 50.2793488293372]}]
    
    print(sorted_configurations[0:args.limit])

    print("Configuration: ", sorted_configurations)
    num_mc = args.num_mc

    output_path_og = args.output_path

    for consistency_factor in args.consistency_factors:
        for temperature in args.temperatures:
            print(temperature)
            args.temperature = float(temperature)
            if args.temperature > 0:
                args.consistency_factor = consistency_factor // args.num_sequences

            else:
                args.consistency_factor = consistency_factor
                
            for i in range(num_mc):
                args.output_path = f"{output_path_og}/{i+1}"

                if not os.path.exists(f"{args.output_path}"):
                    os.mkdir(f"{args.output_path}")
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
