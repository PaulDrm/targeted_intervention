import argparse
import heapq
import pandas as pd
import os
import torch
import numpy as np

import ast

import json


from utils.ut_intervention_utils import (run_llama_intervention_batch_parallel,
                                   get_com_directions,
                                   get_interventions_dict_variable_alpha
)

from utils.ut_processing_utils import (load_model,
                                select_device, 
                                prepare_test_set,
                                process_data,
                                get_fold_indices,
                                get_best_head_metrics_seeds,
                                ParseListOfLists
)

from utils.ut_evaluation_utils import evaluate_configuration, append_to_log_file

def optimize_alpha_for_heads(
    args,
    tokenizer,
    model,
    heads, 
    alphas,
    memoization,
    log,
    log_filename,
    optimized_alphas,
    best_solution
    ):
    

    best_alphas = None
    best_precision = 0
    best_recall = 0
    best_metrics = {"precision": 0, "recall": 0}  
    alpha_step = 0.25 #0.1  # Initialize as a relative step size (10%)
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
        
        #config_key = tuple(sorted(zip(heads, alphas)))
        
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
        precision_consistency = round(np.mean([entry["precision_consistency"] for entry in precision_scores]), 2)
        recall_consistency = round(np.mean([entry["recall_consistency"] for entry in recall_scores]), 2)

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
            "precision_consistency": precision_consistency,
            "recall_consistency": recall_consistency,
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

        if precision < 1 and recall < best_solution["recall"]*0.5:
            print(best_solution['precision'])
            print("Precision and recall too low. Stopping optimization.")
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

def add_to_current_heads(current_heads, next_head):
    # Check if next_head is a single head (tuple of integers)
    # or multiple heads (tuple of tuples)
    if isinstance(next_head[0], str):
        # Single head case - next_head is like (0, 1)
        new_heads = list(current_heads) + [next_head]
    else:
        # Multiple heads case - next_head is like ((0, 1), (2, 3))
        print("Multiple head case")
        print(current_heads)
        print(next_head)
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

    optimized_alphas = {}  # Store optimized alphas for each head
    log_filename_complete = f"{args.output_path}/{log_filename}"
    print(f"Loading memoization and log from {log_filename}")
    memoization = {}
    if os.path.exists(log_filename_complete):

        try:

            #log_df = pd.read_csv(log_filename_complete)

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
                        'seeds': None,
                        'alphas': None,
                        'precision': None,
                        'recall':None
                    }

                #print(heads_frozenset)
                #print(log[0:5])
                best_config = get_best_head_metrics_seeds(log, heads_frozenset)

                print(best_config['seeds'])

                #for seed in best_config["seeds"]:
                #    memoization[heads_frozenset]['seeds'].add(seed)

                memoization[heads_frozenset]['seeds'] = best_config['seeds']

                memoization[heads_frozenset]['alphas'] = best_config["alphas"]  # Assume the best alphas per head combination
                # memoization[heads_frozenset]['results'].append({
                #     'precision': best_config["precision"],
                #     'recall': best_config["recall"],
                #     'seed': best_config["seeds"]
                # })

                print(memoization[heads_frozenset]['seeds'])

                memoization[heads_frozenset]['precision'] = best_config["precision"]
                memoization[heads_frozenset]['recall'] = best_config["recall"]

                optimized_alphas[heads_frozenset] = {'alphas': best_config["alphas"], 
                                                     'precision': best_config["precision"],
                                                     'recall': best_config["recall"]}
                

            print(f"Loaded {len(log)} configurations from the log file.")
            
            #print(optimized_alphas)
    
            
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

    sorted_heads = dict(sorted(
            memoization.items(),
            key=lambda x: (
                len(x[1]["seeds"]),       # Prefer entries with more seeds (primary sort)
                x[1]["precision"],        # Then by precision (secondary sort)
                x[1]["recall"],           # Then by recall (tertiary sort)
                -x[1]["alphas"][0]        # Finally, prefer lower alphas (quaternary sort, negate for ascending order)
            ),
            reverse=True  # Sort in descending order for the primary criteria
        ))

    # Get an iterator over the dictionary items
    first_item = next(iter(sorted_heads.items()))
    # Unpack the first key-value pair
    heads, config = first_item
    # Initialize best solution
    best_solution = {"heads": heads, "alphas": config['alphas'], "precision": config['precision'], "recall": config['recall']}
    # Update best solution from loaded log if available
    min_recall = best_solution["recall"]*0.4 #.values[0]

    counter = 0 
    for count, key in enumerate(list(memoization.keys())): 

        if not all(seed in memoization[key]['seeds'] for seed in args.seeds):
            
            #print(key, "does not contain all seeds.")

            seeds = args.seeds

            best_config = memoization[key]#['results']
        
            #print(best_config)
            #print(best_config["precision"])
            #print(best_config["recall"])

            if best_config['precision'] >= min_precision and best_config['recall'] >= min_recall:
                    counter += 1
    
    print("COUNTER: ", counter) 
    for count, key in enumerate(list(sorted_heads.keys())): 

        if not all(seed in memoization[key]['seeds'] for seed in args.seeds):
            
            print(key, "does not contain all seeds.")

            seeds = args.seeds

            best_config = memoization[key]#['results']

            if best_config['precision'] >= min_precision and best_config['recall'] >= min_recall:

                #args.seeds = list(set(seeds) - set(memoization[key]['seeds']))
                # config = {
                #     "heads": key,
                #     "alphas": memoization[key]["alphas"],
                #     "precision": memoization[key]["precision"],
                #     "recall": memoization[key]["recall"],
                # }
                args.seeds = seeds
                heads_tuple = tuple(key)
                new_alphas = memoization[key]["alphas"]
                original_recall = memoization[key]["recall"]
                original_precision = memoization[key]["precision"]

                # Optimize alphas for the combined configuration
                best_alphas, metrics = optimize_alpha_for_heads(
                    heads=heads_tuple,
                    alphas=new_alphas,
                    args=args,
                    tokenizer=tokenizer,
                    model=model,
                    memoization=memoization,
                    log=log,
                    log_filename=log_filename,
                    optimized_alphas=optimized_alphas,
                    best_solution = {'precision': 0 , 'recall': 0},
                )

                mean_precision = metrics['precision']
                mean_recall = metrics['recall']
                mean_alphas = best_alphas
                
                # mean_precision = np.mean([metrics['precision'], original_precision])
                # mean_recall = np.mean([metrics['recall'], original_recall])
                # mean_alphas = [float(np.mean([alpha1,alpha2])) for alpha1,alpha2 in zip(new_alphas, best_alphas)]

                memoization[heads_tuple] = {
                    'seeds': seeds,
                    'alphas': mean_alphas,
                    'precision': mean_precision,
                    'recall':mean_recall,
                    }
        
                # for seed in seeds:
                #     memoization[heads_tuple]['seeds'].add(seed)
                
                log_entry = {
                    "heads": heads_tuple,
                    'alphas': mean_alphas,
                    'precision': mean_precision,
                    'recall':mean_recall,
                    "seeds": seeds,
                    
                }
                print(f"Added entry to log: {log_entry}")

                args.seeds = seeds

                append_to_log_file(args, log_filename, log_entry)
                #memoization[heads_tuple]['alphas'] = best_alphas
                #memoization[heads_tuple]['precision'] = metrics['precision']
                #memoization[heads_tuple]['recall'] = metrics['recall']

    #data = pd.DataFrame(log)
    # Create the DataFrame
    rows = []
    for key, values in memoization.items():

        row = {
            'heads': key, 
            'seeds': list(values['seeds']), 
            'alphas': values['alphas'], 
            'precision': values['precision'], 
            'recall': values['recall']
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    #print(df)
    if not df.empty:

        print("Reading log data...")
        sorted_df = df[(df['precision'] == 1) 
                         & (df['seeds'].apply(lambda seed_list: 5678 in seed_list))
        ].sort_values(by='recall', ascending=False)#.head(10)

        min_recall = sorted_df.iloc[0].recall*0.4 #.values[0]

        filtered_df = sorted_df[sorted_df['recall'] >= min_recall]
        print(filtered_df)

        for row in filtered_df.itertuples():
            qid = 1
            best_config = memoization[row.heads]

            config = {
                        "heads": row.heads,
                        "alphas": best_config["alphas"],
                        "precision": best_config["precision"],
                        "recall": best_config["recall"],
                    }
            
            heapq.heappush(
                        queue,
                        (
                            -best_config["precision"],
                            -best_config["recall"],
                            len(config["heads"]),
                            sum(config["alphas"]),
                            qid,
                            config
                        ),
                    )

        print(len(queue))
    
    
    # Initialize best solution
    best_solution = {"heads": [], "alphas": [], "precision": 0, "recall": 0}
    # Update best solution from loaded log if available
    
    if log:
        best_entry = max(log, key=lambda x: (len(x["seeds"]), x["precision"], x["recall"]))
        best_solution = {
            "heads": best_entry["heads"],
            "alphas": best_entry["alphas"],
            "precision": best_entry["precision"],
            "recall": best_entry["recall"],
            "seeds": best_entry["seeds"],
        }
        min_recall = best_entry["recall"]*0.4 #.values[0]

    #print(min_recall )
    while queue and iterations < max_iterations:
        
        iterations += 1
        # Get the current best configuration
        _, _, _, _, id, config = heapq.heappop(queue)
        
        current_heads = config["heads"]
        current_alphas = config["alphas"]
        print("Configuration heads: ", config["heads"])
        print("Configuration alphas: ", config["alphas"])
        #print("Queue: ", queue)

        #print(memoization)
        sorted_heads = dict(sorted(
            memoization.items(),
            key=lambda x: (
                len(x[1]["seeds"]),       # Prefer entries with more seeds (primary sort)
                x[1]["precision"],        # Then by precision (secondary sort)
                x[1]["recall"],           # Then by recall (tertiary sort)
                -x[1]["alphas"][0]        # Finally, prefer lower alphas (quaternary sort, negate for ascending order)
            ),
            reverse=True  # Sort in descending order for the primary criteria
        ))
        
        print("Sorted_heads: ", list(sorted_heads.keys())[0:3])
        print("Min recall: ", min_recall)

        new_configuration_detected = False  # Initialize the flag

        for head in sorted_heads.keys():
            
            #if head not in current_heads:
            if all(item not in current_heads for item in head):
                # Check if the head meets minimum performance criteria
                best_config = memoization[head]#['results']

                #print(best_config["precision"])
                #print(best_config["recall"])

                #print(best_config['seeds'] == list(args.seeds))
                #print(args.seeds)
                #print(best_config['seeds'])
                if best_config['precision'] >= min_precision and best_config['recall'] >= min_recall and best_config['seeds'] == list(args.seeds):
                        
                    print(head)
                    print(best_config)
                    
                    new_heads = add_to_current_heads(current_heads, head)
                    
                    # #new_combination = frozenset(new_heads)
                    # # Calculate original and new lengths
                    # current_length = len(current_alphas)
                    # memoization_length = len(memoization[head]['alphas'])
                    # new_total_length = current_length + memoization_length
                    
                    # current_alphas = [alpha/ new_total_length * current_length for alpha in current_alphas]

                    # added_alphas = [alpha/ new_total_length * memoization_length for alpha in memoization[head]['alphas']]

                    # new_alphas = current_alphas + added_alphas

                    new_alphas =current_alphas  + memoization[head]['alphas']

                    new_alphas = [alpha*0.7 for alpha in new_alphas]
                    
                    sorted_pairs = sorted(zip(new_heads, new_alphas))

                    new_heads, new_alphas = zip(*sorted_pairs)
                    
                    heads_tuple = tuple(new_heads)

                    if not heads_tuple in list(memoization.keys()):
                        new_configuration_detected = True  # Mark new configuration as detected
                        break
                    else:
                        print("Configuration already exists in memoization: ", new_heads)
                        #break


        print("New configuration detected: ", new_configuration_detected)
        if not new_configuration_detected:
            continue

        print("Configuration heads: ", heads_tuple)
        print("Configuration initial alphas: ", new_alphas)

        # Optimize alphas for the combined configuration
        best_alphas, metrics = optimize_alpha_for_heads(
            heads=heads_tuple,
            alphas=new_alphas,
            args=args,
            tokenizer=tokenizer,
            model=model,
            memoization=memoization,
            log=log,
            log_filename=log_filename,
            optimized_alphas=optimized_alphas,
            best_solution=best_solution,
        )
        
        memoization[heads_tuple] = {
                    'seeds': set(),
                    'alphas': None,
                    'precision': None,
                    'recall':None
                    }
        
        for seed in args.seeds:
            memoization[heads_tuple]['seeds'].add(seed)
        
        memoization[heads_tuple]['alphas'] = best_alphas
        memoization[heads_tuple]['precision'] = metrics['precision']
        memoization[heads_tuple]['recall'] = metrics['recall']

        #if 
        # Store the result
        optimized_alphas[heads_tuple] = {
            'heads': heads_tuple,
            'alphas': best_alphas,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
        }

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

            min_recall = metrics["recall"]*0.4

        else:
            # If combining with best head didn't improve, put the original config back in queue
            # with slightly lower priority to try other combinations next time
            heapq.heappush(
                queue,
                (-config["precision"], -config["recall"], len(current_heads), sum(current_alphas), id + 2, config)
            )

        # Optional: Print status for debugging
        print(f"Tried combination: {current_heads} + {head}")
        print(f"Performance improved: {performance_improved}")
        
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
    parser.add_argument("--consistency_factor", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="[1234,5678,9012]", help="seeds as a JSON array")
    parser.add_argument("--num_sequences", type=int, default=2)

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
    #log_filename = "configuration_log.csv"
    
    log = []
    
    log_filename = "configuration_optimization.csv"
    
    #log_filename_input = "configuration_optimization.csv"
    #log_filename = "configuration_optimization_multi_fidelity.csv"
    #log_path = f"{args.output_path}/{log_filename}"

    candidate_heads = args.list_of_heads  # List of candidate attention heads
    initial_alpha = 20#args.alpha  # Starting alpha value
    alpha_step = 5
    precision_threshold = 0.05  # Define the precision threshold
    min_precision = .90  # Minimum precision for a head to be considered effective
    min_recall = .10  # Minimum recall for a head to be considered effective

    device = select_device(min_vram_gb=20)

    tokenizer, model = load_model(args.model_name, device)

    num_heads = 32# args.num_heads  # From args

    #args.seeds =[1234, 5678, 9012]

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
