import argparse
import heapq
import pandas as pd
import os
import torch
import numpy as np

import json

from ut_intervention_utils import (run_llama_intervention_batch_parallel,
                                   get_com_directions,
                                   get_interventions_dict_variable_alpha
)

from ut_processing_utils import (load_model,
                                select_device, 
                                prepare_test_set,
                                process_data,
                                get_com_directions,
                                get_fold_indices,
                                ParseListOfLists
)

from ut_evaluation_utils import (precision_recall_consistency)

def evaluate_configuration(args, tokenizer, model, heads, alphas, num_heads):
    # print(args.input_path)
    df = pd.read_json(args.input_path)

    #print(df.columns)
    #print(df.shape)
    # Set the appropriate id_column based on the dataset
    id_column = "data_id" if args.dataset_name != "requirements_data" else "req_id"
    num_heads =32
    precision_scores = []
    recall_scores = []

    # Run for multiple seeds
    for seed in args.seeds:
        args.seed = seed
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Process the data
        index_dic, separated_activations, separated_labels, reqs_order = process_data(df, id_column)

        # Create folds
        number_of_examples = np.arange(len(reqs_order)) 
        
        #print(args.num_fold)
        fold_idxs = np.array_split(number_of_examples, args.num_fold)

        # Run the experiment for each fold
        for fold_index in range(args.num_fold):

            # Determine train, validation, and test sets
            train_idxs, val_set_idxs, test_idxs = get_fold_indices(fold_index, args, fold_idxs, reqs_order)

            # Get the top heads
            top_heads= heads# args.list_of_heads

            print("Heads intervened: ", sorted(top_heads))
            
            # Get the interventions
            tuning_activations = np.concatenate(separated_activations, axis=0)
            com_directions = get_com_directions(32, 32, train_idxs, val_set_idxs, separated_activations, separated_labels) if args.use_center_of_mass else None
            
            print(f"Evaluating configuration for head {heads} with alphas {alphas} for seed {args.seed} and fold {fold_index}")
            # Prepare the interventions dictionary
            interventions = get_interventions_dict_variable_alpha(
                heads, alphas, tuning_activations, num_heads, args.use_center_of_mass,
                args.use_random_dir, com_directions
            )
            test_index_expanded = np.concatenate([list(index_dic.values())[i] for i in test_idxs])
        
            # Run the intervention
            test_set = df.loc[test_index_expanded]
            #print(test_set.shape)
            
            args.consistency_factor = 5
            test_set = prepare_test_set(test_set, args)
            #print(test_set.shape)

            test_set = test_set.iloc[0:int(test_set.shape[0])]
            
            #results = run_llama_intervention_batch(args, tokenizer, model, interventions, test_set)
            results = run_llama_intervention_batch_parallel(args, tokenizer, model, interventions, test_set)

            curr_fold_results = pd.DataFrame(results)

            # Calculate precision and recall
            precision, recall = precision_recall_consistency(curr_fold_results)

            # Store the results
            precision_scores.append({"precision": precision, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas})
            recall_scores.append({"recall": recall, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas})

            if args.head== None:
                curr_fold_results.to_json(f"{args.output_path}/results_test_intervention_seed_{int(args.seed)}_{int(args.alpha)}_{args.layer}.json", orient='records', indent=4)

            else: 
                curr_fold_results.to_json(f"{args.output_path}/results_test_intervention_seed_{int(args.seed)}_{int(args.alpha)}_{args.layer}_{args.head}.json", orient='records', indent=4)

            value_counts = curr_fold_results.predict.value_counts().to_dict()
            value_counts['alpha'] = args.alpha
            value_counts['layer'] = args.layer
            value_counts['seed'] = args.seed
            
            if args.head != None: 
                value_counts['head'] = args.head

            # Path to your JSON file
            json_file_path = f'{args.output_path}/overall_results.json'
            
            # Load existing data from the JSON file
            try:
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
            except FileNotFoundError:
                data = []

            # Write the updated data back to the JSON file
            with open(json_file_path, 'w') as file:
                json.dump(data, file, indent=4)

    return precision_scores, recall_scores

def optimize_alphas_for_head(
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
    alpha_step = 10 / len(heads)  # Initial step size
    direction = 1  # 1 for increasing, -1 for decreasing
    max_iterations = 5  # Prevent infinite loop
    
    for _ in range(max_iterations):
        
        #print("Current heads:", heads)
        #print("Current alphas:", alphas)
        #heads = [tuple(head) for head in heads]
        config_key = tuple(sorted(zip(heads, alphas)))
        #print(config_key)
        #config_key = tuple((tuple(inner_tuple[0]), inner_tuple[1]) for inner_tuple in config_key)
        #print(config_key)
        
        if config_key in memoization:
            continue

        precision_scores, recall_scores = evaluate_configuration(
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
            best_alphas = alphas #.copy()
            best_heads = heads #.copy()
            best_precision = precision
            best_recall = recall
            best_metrics = {"precision": precision, "recall": recall}
            print("Heads for logging: ", heads)
            print("Best alphas:", best_alphas)
            print("Best metrics:", best_metrics)
        
        # Log and memoize the configuration
        log_entry = {
            "heads": heads,
            "alphas": [float(alpha) for alpha in alphas],
            "precision": precision,
            "recall": recall,
        }

        log.append(log_entry)
        append_to_log_file(log_filename, log_entry)
        memoization.add(config_key)

        # Adjust alphas based on precision
        if precision < 1:
            if direction == -1:
                alpha_step /= 2  # Reduce step size when changing direction
            direction = 1
            alphas = [alpha + alpha_step for alpha in alphas]
        
            print("Increasing alphas:", alphas)
        
        elif precision == 1:
            if direction == 1:
                alpha_step /= 2  # Reduce step size when changing direction
            direction = -1
            alphas = [alpha - alpha_step for alpha in alphas]
        
            print("Decreasing alphas:", alphas)
        
        else:
        
            print("Problem: Unexpected precision value!")
            break

        # Ensure alphas stay within a reasonable range
        alphas = [max(0, min(alpha, 100)) for alpha in alphas]

        # Stop if the step size becomes too small
        if alpha_step < 1:
            break

    # Update optimized_alphas dictionary with all heads and their corresponding best alphas
    #print(heads)
    #print(best_alphas)
    if best_alphas != None:
        optimized_alphas[tuple(best_heads)] = {
            "alphas": best_alphas,
            "precision": best_precision,
            "recall": best_recall
        }
    return best_alphas, best_metrics


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
    max_iterations = 8  # Set a reasonable upper limit to prevent infinite loops
    no_improve_counter = 0  # Counter for iterations without improvement
    required_no_improve = 3  # Number of consecutive non-improving iterations to trigger stopping

    iteration = 0

    while iteration < max_iterations and no_improve_counter < required_no_improve:
        iteration += 1
        print(f"--- Iteration {iteration} ---")
        print("Current heads:", heads)
        print("Current alphas:", alphas)
        
        config_key = tuple(sorted(zip(heads, alphas)))
        
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

def append_to_log_file(filename, log_entry):
    # Convert log_entry to a DataFrame and append to CSV
    log_df = pd.DataFrame([log_entry])
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        log_df.to_csv(filename, mode="w", index=False)
    else:
        log_df.to_csv(filename, mode="a", index=False, header=False)

def get_best_head_metrics(log, head):
    """
    Extract the best configuration for a specific head based on:
    1. Highest precision
    2. If precision ties, highest recall
    3. If both tie, lowest alpha
    
    Args:
        log (list): List of dictionaries containing head configurations and metrics
        head (tuple): The head configuration to search for
    
    Returns:
        dict: Best configuration for the head or None if not found
    """
    # Filter entries for the specific head
    head_entries = [
        entry for entry in log 
        if len(entry["heads"]) == 1 and entry["heads"][0] == head
    ]
    
    if not head_entries:
        return None


    # Sort by our criteria
    best_entry = max(
        head_entries,
        key=lambda x: (
            x["precision"],
            x["recall"],
            -x["alphas"][0]  # Negative so lower alpha is preferred
        )
    )
    
    return best_entry

def add_to_current_heads(current_heads, next_head):
    # Check if next_head is a single head (tuple of integers)
    # or multiple heads (tuple of tuples)
    if isinstance(next_head[0], str):
        # Single head case - next_head is like (0, 1)
        new_heads = list(current_heads) + [next_head]
    else:
        # Multiple heads case - next_head is like ((0, 1), (2, 3))
        print("Multiple head case")
        new_heads = list(current_heads) + list(next_head)
        print(frozenset(new_heads))
    #print("Current heads:", current_heads)
    #print("Next head to add:", next_head)
    #print("Resulting combination:", new_heads)
    
    return new_heads

def prune_and_branch_per_head_alpha(
    candidate_heads,
    initial_alpha,
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
    ineffective_heads = set()  # Heads that are ineffective individually
    pruned_heads = set()  # Heads pruned from further consideration
    iterations = 0

    print(f"Candidate heads: {candidate_heads}")

    #candidate_heads = [tuple(head) for head in candidate_heads]  # Convert set to list for easier manipulation

    optimized_alphas = {}  # Store optimized alphas for each head

    # Load memoization and log from existing log file if it exists
    if os.path.exists(log_filename):

        try:
            log_df = pd.read_csv(log_filename)
            for _, row in log_df.iterrows():
                heads = eval(row["heads"]) if isinstance(row["heads"], str) else row["heads"]
                alphas = eval(row["alphas"]) if isinstance(row["alphas"], str) else row["alphas"]
                config_key = tuple(sorted(zip(heads, alphas)))
                # Convert the inner list to a tuple
                #config_key = tuple((tuple(inner_tuple[0]), inner_tuple[1]) for inner_tuple in config_key)
                
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
            print(f"File '{log_filename}' is either empty or has no columns to parse.")
            # Debugging: show the raw content of the file
            with open(log_filename, 'r') as file:
                content = file.read().strip()
                if not content:
                    print(f"'{log_filename}' is completely empty.")
                else:
                    print(f"Content of the file:\n{content}")
        
        
    else:
        print("No existing log file found. Starting fresh.")

    # Evaluate each head individually with alpha optimization
    for id, head in enumerate(candidate_heads):
        
        # Usage in your main loop:
        if any(len(entry["heads"]) == 1 and entry["heads"][0] == tuple(head) for entry in log):
            print(f"Head {head} has already been evaluated individually.")
            
            # Get best configuration for this head
            best_config = get_best_head_metrics(log, tuple(head))
            
            if best_config["precision"] < min_precision or best_config["recall"] < min_recall:
                ineffective_heads.add(tuple(head))
            
            else:
                
                # Add to the queue for further exploration
                config = {
                    "heads": [tuple(head)],
                    "alphas": [best_config["alphas"][0]],
                    "precision": best_config["precision"],
                    "recall": best_config["recall"],
                }
                #print("Push to queue:")
                #print(config)
                
                heapq.heappush(
                    queue,
                    (
                        -best_config["precision"],
                        -best_config["recall"],
                        len(config["heads"]),
                        sum(config["alphas"]),
                        id,
                        config
                    ),
                )
            continue  # Skip re-evaluating the head
        
        # Zip the heads and alphas, then sort them
        heads = [tuple(head) for head in [head]]
        sorted_pairs = sorted(zip(heads, [initial_alpha]))
        heads, alphas = zip(*sorted_pairs)
        
        print("Heads before entering function", heads)
        best_alphas, metrics = optimize_alphas_for_head(
            heads=heads,
            alphas=alphas,
            args=args,
            tokenizer=tokenizer,
            model=model,
            num_heads=num_heads,
            memoization=memoization,
            log=log,
            log_filename=log_filename,
            optimized_alphas=optimized_alphas,
        )

        heads = [tuple(head) for head in [head]]
        best_alphas = [float(alpha) for alpha in best_alphas]
        
        optimized_alphas[tuple(heads)] = {
                            "alphas": best_alphas,
                            "precision": metrics['precision'],
                            "recall": metrics['recall']
                        }

        # Check if the head meets minimum performance criteria
        if metrics["precision"] >= min_precision and metrics["recall"] >= min_recall:
            # Add to the queue for further exploration
            config = {
                "heads": [tuple(head)],
                "alphas": best_alphas,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
            heapq.heappush(
                queue, (-metrics["precision"], -metrics["recall"], len(config["heads"]), sum(config['alphas']), id, config)
            )
        else:
            # Mark the head as ineffective
            ineffective_heads.add(tuple(head))

    # Remove ineffective heads from candidate_heads
    candidate_heads = [tuple(head) for head in candidate_heads if tuple(head) not in ineffective_heads]

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

    # Add at the start of your script, outside the loop:
    tried_head_combinations = set()  # To track which combinations have already been attempted

    while queue and iterations < max_iterations:
        
        iterations += 1
        # Get the current best configuration
        _, _, _, _, id, config = heapq.heappop(queue)
        
        current_heads = config["heads"]
        current_alphas = config["alphas"]
        print(config["heads"])
        print("Queue: ", queue)
        tried_head_combinations.add(frozenset(current_heads))
        print(tried_head_combinations)

        # Get all historical performance from optimized_alphas
        head_performance = {}
        for head, data in optimized_alphas.items():
            if head not in current_heads and head not in pruned_heads:
                
                # Check if we've already tried this combination
                #potential_combination = frozenset(list(current_heads) + [head])
                print(head)
                #if isinstance(head[0], str):
                    
                potential_combination = add_to_current_heads(current_heads, head)
                potential_combination = frozenset(potential_combination)
                
                if potential_combination not in tried_head_combinations:
                
                    precision = data.get("precision")
                    recall = data.get("recall")
                    head_performance[head] = (precision, recall)
    
        #print("Head performance: ", head_performance)

        # Sort all heads by their historical performance
        sorted_heads = sorted(
            head_performance.keys(),
            key=lambda h: (head_performance[h][0], head_performance[h][1]),
            reverse=True
        )
        
        #print("Next best head: ", sorted_heads)

        # If no historical heads available, continue to next iteration
        if not sorted_heads:
            continue
            
        # Get the next best head not in current configuration
        next_head = sorted_heads[0]
       
        # Record this combination as tried
        # Create new combination and add to tried combinations
        new_heads = add_to_current_heads(current_heads, next_head)
        new_combination = frozenset(new_heads)
        
        tried_head_combinations.add(new_combination)

        # Try combining with the single best head
        total_heads = len(new_combination) #len(current_heads) + 1
        adjusted_alphas = [alpha * len(current_heads) / total_heads for alpha in current_alphas]
        
        # Get historical alpha for the new head
        historical_alphas = optimized_alphas.get(next_head, {"alphas": [initial_alpha]})["alphas"]#[0]
        reduced_alpha_values = [float(historical_alpha) / total_heads for historical_alpha in historical_alphas]
        
        # Combine heads and alphas
        #new_heads = list(current_heads) + [next_head]
        new_alphas = adjusted_alphas + reduced_alpha_values
        
        # Convert heads to tuples and sort
        new_heads = [tuple(head) for head in new_heads]

        print("New heads: ", new_heads)
        print("New alphas: ", new_alphas)

        sorted_pairs = sorted(zip(new_heads, new_alphas))

        new_heads, new_alphas = zip(*sorted_pairs)
        
        # Optimize alphas for the combined configuration
        best_alphas, metrics = optimize_alphas_for_head(
            heads=new_heads,
            alphas=new_alphas,
            args=args,
            tokenizer=tokenizer,
            model=model,
            num_heads=num_heads,
            memoization=memoization,
            log=log,
            log_filename=log_filename,
            optimized_alphas=optimized_alphas,
        )
        
        # Check if performance improved
        performance_improved = False
        if metrics["precision"] > best_solution['precision']:
            performance_improved = True
        elif abs(metrics["precision"] - best_solution['precision']) <= precision_threshold and metrics[
            "recall"
        ] > best_solution['recall']:
            performance_improved = True
        elif (
            abs(metrics["precision"] - best_solution['precision']) <= precision_threshold
            and abs(metrics["recall"] - best_solution['recall']) <= precision_threshold
            and len(new_heads) < len(best_solution["heads"])
        ):
            performance_improved = True
        
        if performance_improved:
            # Update best solution and add to queue
            new_config = {
                "heads": new_heads,
                "alphas": best_alphas,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
            
            best_solution = {
                "heads": new_heads,
                "alphas": best_alphas,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
            
            heapq.heappush(
                queue, 
                (-metrics["precision"], -metrics["recall"], len(new_heads), sum(best_alphas), id + 1, new_config)
            )

        else:
            # If combining with best head didn't improve, put the original config back in queue
            # with slightly lower priority to try other combinations next time
            heapq.heappush(
                queue,
                (-config["precision"], -config["recall"], len(current_heads), sum(current_alphas), id + 2, config)
            )

        # Optional: Print status for debugging
        print(f"Tried combination: {current_heads} + {next_head}")
        print(f"Performance improved: {performance_improved}")
        print(f"Total combinations tried: {len(tried_head_combinations)}")

    # Save the log to a file
    #log_df = pd.DataFrame(log)
    #log_df.to_csv(log_filename, index=False)

    return best_solution, log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama_7B", help="model name")
    parser.add_argument("--num_heads", type=int, default=48, help="K, number of top heads to intervene on")
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
    parser.add_argument("--layer", type=int, help="layer for intervention")
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
    parser.add_argument("--list_of_heads", type=str, default="", action=ParseListOfLists, help="Input should be a list of lists (e.g., [['11', '0'], ['13', '0']]).")
    parser.add_argument("--prompt_type", type=str, default="ab_cot")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--consistency_factor", type=int, default=6)

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
    
    candidate_heads = args.list_of_heads  # List of candidate attention heads
    initial_alpha = 20#args.alpha  # Starting alpha value
    alpha_step = 5
    precision_threshold = 5.0  # Define the precision threshold
    min_precision = .90  # Minimum precision for a head to be considered effective
    min_recall = .10  # Minimum recall for a head to be considered effective

    device = select_device(min_vram_gb=20)

    tokenizer, model = load_model(args.model_name, device)

    num_heads = 32# args.num_heads  # From args

    args.seeds =[1234, 5678, 9012]

    args.add_or_subtract = False

    best_configuration, log = prune_and_branch_per_head_alpha(
        candidate_heads=candidate_heads,
        initial_alpha=initial_alpha,
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
