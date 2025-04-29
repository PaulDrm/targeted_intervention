import argparse
import heapq
import pandas as pd
import os
import torch
import numpy as np

import json

from ut_evaluation_utils import evaluate_configuration, append_to_log_file

from ut_processing_utils import (load_model,
                                select_device, 
                                prepare_test_set,
                                process_data,
                                get_fold_indices,
                                get_best_head_metrics,
                                ParseListOfLists
)


from ut_intervention_utils import (run_llama_intervention_batch_parallel,
                                   get_com_directions,
                                   get_interventions_dict_variable_alpha
)

import numpy as np

# def optimize_alpha_for_heads(
#     args,
#     tokenizer,
#     model,
#     heads, 
#     alphas,
#     num_heads,
#     memoization,
#     log,
#     log_filename,
#     optimized_alphas,
# ):

#     best_alphas = None
#     best_precision = 0
#     best_recall = 0
#     best_metrics = {"precision": 0, "recall": 0}  
#     #alpha_step = 20 / len(heads)  # Initial step size
#     alpha_step = 2
#     direction = 1  # 1 for increasing, -1 for decreasing
#     #max_iterations = 5  # Prevent infinite loop
#     max_iterations = 3
#     for _ in range(max_iterations):
        
#         print("Current heads:", heads)
#         print("Current alphas:", alphas)
#         #heads = [tuple(head) for head in heads]
#         config_key = tuple(sorted(zip(heads, alphas)))
#         #print(config_key)
#         #config_key = tuple((tuple(inner_tuple[0]), inner_tuple[1]) for inner_tuple in config_key)
#         #print(config_key)
        
#         if config_key in memoization:
#             print("Using memoized results for configuration:", config_key)
#             continue
        
#         precision_scores, recall_scores, undefined = evaluate_configuration(
#             args=args,
#             tokenizer=tokenizer,
#             model=model,
#             heads=heads,
#             alphas=alphas,
#             num_heads=num_heads,
#         )
        
#         precision = round(np.mean([entry["precision"] for entry in precision_scores]), 2)
#         recall = round(np.mean([entry["recall"] for entry in recall_scores]), 2)

#         # Update best configuration if performance improves
#         if precision > best_precision or (precision == best_precision and recall > best_recall):
#             print("New best configuration found!")
#             best_alphas = alphas #.copy()
#             best_heads = heads #.copy()
#             best_precision = precision
#             best_recall = recall
#             best_metrics = {"precision": precision, "recall": recall}
#             print("Heads for logging: ", heads)
#             print("Best alphas:", best_alphas)
#             print("Best metrics:", best_metrics)
        

#         # Log and memoize the configuration
#         log_entry = {
#             "heads": heads,
#             "alphas": [float(alpha) for alpha in alphas],
#             "precision": precision,
#             "recall": recall,
#         }


#         log.append(log_entry)
#         append_to_log_file(args, log_filename, log_entry)
        
#         # Adjust alphas based on precision

#         if precision == 1 or undefined == 1: 

#             if direction == 1:
#                 alpha_step *= 0.65  # Reduce step size when changing direction
#             else: 
#                 alpha_step *=1.15
#             direction = -1
#             #alphas = [alpha - alpha_step for alpha in alphas]
#             alphas = [alpha * (1 - alpha_step) for alpha in alphas]
#             print("Decreasing alphas:", alphas)
        
#         elif precision < 1:
#             if direction == -1:
#                 alpha_step *= 0.65  # Reduce step size when changing direction
#             else: 
#                 alpha_step *=1.15
#             direction = 1
#             #alphas = [alpha + alpha_step for alpha in alphas]
#             alphas = [alpha * (1 + alpha_step) for alpha in alphas]
#             print("Increasing alphas:", alphas)
            
#         else:
        
#             print("Problem: Unexpected precision value!")
#             break

#         # Ensure alphas stay within a reasonable range
#         alphas = [max(0, min(alpha, 100)) for alpha in alphas]

#         # Stop if the step size becomes too small
#         if alpha_step < 1/len(heads):
#             break

#     # Update optimized_alphas dictionary with all heads and their corresponding best alphas
#     #print(heads)
#     #print(best_alphas)

#     if best_alphas != None:
#         optimized_alphas[tuple(best_heads)] = {
#             "alphas": best_alphas,
#             "precision": best_precision,
#             "recall": best_recall
#         }

#     return best_alphas, best_metrics


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
    import numpy as np  # Ensure numpy is imported

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
        
        config_key = tuple(sorted(zip(heads, alphas)))
        print(f"Configuration key: {config_key}")
        if config_key in memoization:
            print("Using memoized results for configuration:", config_key)
            # Skip the rest of the loop and continue to the next iteration
            no_improve_counter += 1
            if no_improve_counter >= required_no_improve:
                print("No improvement after using memoized results. Stopping optimization.")
                break
            continue
        
        precision_scores, recall_scores, undefined = evaluate_configuration(
            args=args,
            tokenizer=tokenizer,
            model=model,
            heads=heads,
            alphas=alphas,
            num_heads=num_heads,
        )
        
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
        }

        log.append(log_entry)
        append_to_log_file(args, log_filename, log_entry)
        
        memoization.add(config_key)  # Assuming memoization is a set. If it's a dict, adjust accordingly.

        # Adjust alphas based on precision
        if precision == 1 or undefined == 1: 
            if direction == 1:
                alpha_step *= 0.5  # Reduce step size when changing direction
            else: 
                alpha_step *= 1.15
            direction = -1
            alphas = [alpha * (1 - alpha_step) for alpha in alphas]
            print("Decreasing alphas:", alphas)
        
        elif precision < 1:
            if direction == -1:
                alpha_step *= 0.5  # Reduce step size when changing direction
            else: 
                alpha_step *= 1.15
            direction = 1
            alphas = [alpha * (1 + alpha_step) for alpha in alphas]
            print("Increasing alphas:", alphas)
            
        else:
            print("Problem: Unexpected precision value!")
            break

        # Ensure alphas stay within a reasonable range
        alphas = [max(0, min(alpha, 100)) for alpha in alphas]

        # Stop if the step size becomes too small
        #if alpha_step < 1 / len(heads):
        #    print("Alpha step size too small. Stopping optimization.")
        #    break

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
    log_filename,
    precision_threshold=5.0,
    min_precision=0.0,
    min_recall=0.0,
    max_iterations=1000,
):
    # Initialize the priority queue
    queue = []
    iterations = 0
    overall_best_config = None
    min_heads = 1
    optimized_alphas = {}  # Store optimized alphas for each head

    # Add at the start of your script, outside the loop:
    tried_head_combinations = set()  # To track which combinations have already been attempted


    # Load memoization and log from existing log file if it exists



    log_filename_complete = f"{args.output_path}/{log_filename}"
    print(f"Loading memoization and log from {log_filename}")

    if os.path.exists(log_filename_complete):

        try:

            log_df = pd.read_csv(log_filename_complete)
            for _, row in log_df.iterrows():
                heads = eval(row["heads"]) if isinstance(row["heads"], str) else row["heads"]
                alphas = eval(row["alphas"]) if isinstance(row["alphas"], str) else row["alphas"]
                config_key = tuple(sorted(zip(heads, alphas)))
                # Convert the inner list to a tuple
                #config_key = tuple((tuple(inner_tuple[0]), inner_tuple[1]) for inner_tuple in config_key)
                tried_head_combinations.add(frozenset(heads))
                memoization.add(config_key)
                log_entry = {
                    "heads": heads,
                    "alphas": alphas,
                    "precision": row["precision"],
                    "recall": row["recall"],
                }
                log.append(log_entry)

                # Update optimized_alphas with the best precision/recall for each head
                #if len(heads) == 1:
                #    head = heads[0]
                head_tuple = heads#tuple(head)
                current_precision = row["precision"]
                current_recall = row["recall"]
                current_alphas = alphas

                # Check if we have an existing entry for this head
                if head_tuple in optimized_alphas:
                    existing_precision = optimized_alphas[head_tuple]["precision"]
                    existing_recall = optimized_alphas[head_tuple]["recall"]
                    existing_alphas = optimized_alphas[head_tuple]["alphas"]
                    # Update if current precision is better, or if equal precision and better recall
                    if (current_precision > existing_precision or
                        (current_precision == existing_precision and current_recall > existing_recall) or
                            (current_precision == existing_precision and current_recall == existing_recall and sum(current_alphas) < sum(existing_alphas))):
                        optimized_alphas[head_tuple] = {
                            "alphas": current_alphas,
                            "precision": current_precision,
                            "recall": current_recall
                        }
                else:
                    optimized_alphas[head_tuple] = {
                        "alphas": current_alphas,
                        "precision": current_precision,
                        "recall": current_recall
                    }
            
            print(optimized_alphas)
                    
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
    
    for id, configuration in enumerate(configurations): 
        
        alpha = configuration[1]

        configuration = configuration[0]
        
        if len(configuration) > 1:
            mid = len(configuration) // 2
            first_half = configuration[:mid]
            second_half = configuration[mid:]

        config = {
                "heads": first_half,
                "alphas": [alpha*2]*len(first_half),
                "precision": -1,
                "recall": -0.8,
            }

        # Push both halves to the heap with their priority
        heapq.heappush(queue, (
            -1,  # Negative because heapq is a min-heap
            -0.8,
            len(first_half),  # For tie-breaking
            first_half,
            config))


        config = {
            "heads": second_half,
            "alphas": [alpha*2]*len(second_half),
            "precision": -1,
            "recall": -0.8,
        }

        heapq.heappush(queue, (
            -1,
            -0.8,
            len(second_half),
            second_half,
            config))
        

    checked_configurations = set([entry["heads"] for entry in log])
            
    
    #print(log)

    for configuration in checked_configurations:        

        print("Current configuration:", configuration)
        # Get best configuration for this head
        best_config = get_best_head_metrics(log, configuration)
        
        # Add to the queue for further exploration
        
        print(best_config["precision"])
        print(best_config["recall"])

        if len(configuration) > 1:
            mid = len(configuration) // 2
            first_half = configuration[:mid]
            second_half = configuration[mid:]

            config = {
                "heads": first_half,
                "alphas": [best_config["alphas"][0]]*len(first_half),
                "precision": best_config["precision"],
                "recall": best_config["recall"],
            }

            # Push both halves to the heap with their priority
            heapq.heappush(queue, (
                -best_config["precision"],  # Negative because heapq is a min-heap
                -best_config["recall"],
                len(first_half),  # For tie-breaking
                first_half,
                config))

            config = {
                "heads": second_half,
                "alphas": [best_config["alphas"][0]]*len(second_half),
                "precision": best_config["precision"],
                "recall": best_config["recall"],
            }

            heapq.heappush(queue, (
                -best_config["precision"],  # Negative because heapq is a min-heap
                -best_config["recall"],
                len(second_half),
                second_half,
                config))
            

    # Initialize best solution
    best_solution = {"heads": [], "alphas": [], "precision": 0, "recall": 0}
    # Update best solution from loaded log if available
    
    if log:
        best_entry = max(log, key=lambda x: (x["precision"], x["recall"]))
        
        best_solution = {
            "heads": best_entry["heads"],
            "alphas": best_entry["alphas"],
            "precision": best_entry["precision"],
            "recall": best_entry["recall"],
        }
        
    print("Queue: ", queue)

    
    while queue and iterations < max_iterations:

        iterations += 1
        # Pop the subset with the highest priority (highest precision and recall)
        _, _, _, _, config = heapq.heappop(queue)

        current_heads = config["heads"]
        current_alphas = config["alphas"]
        #print(config["heads"])
        #print("Queue: ", queue)

        heads_tuple = tuple(sorted([tuple(head) for head in current_heads]))

        #print(tried_head_combinations)

        #print("Heads input: ", heads_tuple)
        #print("Memoization: ", memoization)
        # Check if this configuration has already been evaluated
        #if heads_tuple in memoization:
        #    continue
        #print("Memoization: ", tried_head_combinations)

        if frozenset(heads_tuple) in tried_head_combinations:
            print("Skipping configuration: Already evaluated.")
            print(heads_tuple)
            continue

        print(heads_tuple)
        print(len(heads_tuple))
        
        tried_head_combinations.add(frozenset(heads_tuple))

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

        #if 
        # Store the result
        optimized_alphas[heads_tuple] = {
            'heads': heads_tuple,
            'alphas': best_alphas,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
        }

        #memoization.add(heads_tuple)

        # Check if performance is satisfactory
        if metrics['precision'] >= precision_threshold and len(heads_tuple) <= min_heads:
            # Update best configuration if better
            if overall_best_config is None or (
                metrics['precision'] > overall_best_config['precision'] or
                (metrics['precision'] == overall_best_config['precision'] and metrics['recall'] > overall_best_config['recall'])
            ):
                overall_best_config = optimized_alphas[heads_tuple]
        
            
        # Split the heads into two halves and add to heap if size > min_heads
        if len(heads_tuple) > min_heads:
            mid = len(heads_tuple) // 2
            first_half = heads_tuple[:mid]
            second_half = heads_tuple[mid:]

            config = {
                "heads": first_half,
                "alphas": [best_alphas[0]]*len(first_half),
                "precision": metrics['precision'],
                "recall": metrics['recall'],
            }

            # Push both halves to the heap with their priority
            heapq.heappush(queue, (
                -metrics['precision'],
                -metrics['recall'],
                len(first_half),
                first_half,
                config))

            # Push second half
            config_second = {
                "heads": second_half,  # Corrected to second_half
                "alphas": [best_alphas[0]] * len(second_half),
                "precision": metrics['precision'],
                "recall": metrics['recall'],
            }

            heapq.heappush(queue, (
                -metrics['precision'],
                -metrics['recall'],
                len(second_half),
                second_half,
                config_second
            ))
        print(queue)
    return best_config, log

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
    
    parser.add_argument("--input_path", type=str, help="input path")
    parser.add_argument("--output_path", type=str, help="output path")
    parser.add_argument("--dataset_name", type=str, default="requirements_data")
    parser.add_argument(
        "--add_or_subtract",
        type=lambda x: str(x).lower() == "true",
        default="true",
        help="if intervention is added or substract to activations",
    )
    # parser.add_argument(
    #     "--list_of_heads",
    #     type=str,
    #     default="",
    #     help="Input should be a list of lists (e.g., [['11', '0'], ['13', '0']]).",
    # )

    #parser.add_argument("--list_of_heads", type=str, default="", action=ParseListOfLists, help="Input should be a list of lists (e.g., [['11', '0'], ['13', '0']]).")
    parser.add_argument("--prompt_type", type=str, default="ab_cot")
    parser.add_argument("--temperature", type=float, default=0.8)
    #parser.add_argument("--consistency_factor", type=int, default=6)

    args = parser.parse_args()

    if not os.path.exists(f"{args.output_path}"):
        os.mkdir(f"{args.output_path}")

    # Print all arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Initialize variables
    memoization = set()
    log = []
    log_filename = "configuration_log.csv"
    #initial_alpha = 20#args.alpha  # Starting alpha value
    alpha_step = 5
    precision_threshold = 5.0  # Define the precision threshold
    min_precision = .70  # Minimum precision for a head to be considered effective
    min_recall = .10  # Minimum recall for a head to be considered effective
    
    num_heads = 32# args.num_heads  # From args

    args.seeds =[1234, 5678, 9012]

    args.add_or_subtract = False

    # Print all arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    tokenizer, model = load_model(args.model_name)

    log = []
    log_filename = "configuration_log_head_optimization.csv"
    
    log_path = f"{args.output_path}/{log_filename}"
    
    layers = [12,13,15]

    alphas = [1.21875, 2.625, 4.625]

    alphas = [3, 3, 3]

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
        log_filename=log_filename,
        precision_threshold=precision_threshold,
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
