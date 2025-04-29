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

from ut_evaluation_utils import evaluate_configuration

sys.path.append('../app/')

from honest_llama.old_python_scripts.run_experiments import process_data

def append_to_log_file(args, filename, log_entry):
    # Convert log_entry to a DataFrame and append to CSV
    log_df = pd.DataFrame([log_entry])
    filename = f"{args.output_path}/{filename}"
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        log_df.to_csv(filename, mode="w", index=False)
    else:
        log_df.to_csv(filename, mode="a", index=False, header=False)

def optimize_alpha_for_layers(
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
    alpha_step = 20 / len(heads)  # Initial step size
    direction = 1  # 1 for increasing, -1 for decreasing
    max_iterations = 3  # Prevent infinite loop

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
        append_to_log_file(args, log_filename, log_entry)
        
        # Adjust alphas based on precision

        if precision == 1 or undefined == 1: 

            if direction == 1:
                alpha_step *= 0.65  # Reduce step size when changing direction
            
            direction = -1
            alphas = [alpha - alpha_step for alpha in alphas]
        
            print("Decreasing alphas:", alphas)
        
        elif precision < 1:
            if direction == -1:
                alpha_step *= 0.65  # Reduce step size when changing direction
            
            direction = 1
            alphas = [alpha + alpha_step for alpha in alphas]
        
            print("Increasing alphas:", alphas)
            
        else:
        
            print("Problem: Unexpected precision value!")
            break

        # Ensure alphas stay within a reasonable range
        alphas = [max(0, min(alpha, 100)) for alpha in alphas]

        # Stop if the step size becomes too small
        if alpha_step < 1/len(heads):
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


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', help='model name')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.5)
    parser.add_argument('--use_center_of_mass', type=lambda x: (str(x).lower() == 'true'), help='Whether to use the center of mass or not')
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--input_path', type=str, help='input path')
    parser.add_argument('--output_path', type=str, help='output path')
    parser.add_argument('--dataset_name', type=str, default='requirements_data')
    parser.add_argument('--add_or_subtract', type=lambda x: (str(x).lower() == 'true'), default='true', help='if intervention is added or substract to activations')
    parser.add_argument('--test_set_input_path', type=str)
    parser.add_argument('--prompt_type', type=str, default="open_ended")
    parser.add_argument('--add_proj_val_std', type=lambda x: (str(x).lower() == 'true'), default='true')
    parser.add_argument('--temperature', type=float, default=0.8)
    #parser.add_argument('--layer', type=int, help='layer for intervention')
    #parser.add_argument('--head', type=int, help='head for intervention')
    #parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    #parser.add_argument('--device', type=int, default=0, help='device')
    
    args = parser.parse_args()

    if not os.path.exists(f"{args.output_path}"):
        os.mkdir(f"{args.output_path}")

    # Print all arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    seeds = [5678, 9012]  # List of seeds to test

    args.seeds = seeds  # Use the same seeds for all experiments

    tokenizer, model = load_model(args.model_name)

    log = []
    log_filename = "configuration_log_layers.csv"
    
    # Initialize variables
    memoization = set()

    optimized_alphas = {}
    log_path = f"{args.output_path}/{log_filename}"
    if os.path.exists(log_path):

        try:
            log_df = pd.read_csv(log_path)
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

        except:
            print("Error reading log file.")

    num_layers = 32 #model.config.num_hidden_layers
    num_heads = 32 #model.config.num_attention_heads

    for layer in range(12,num_layers):

        args.layer = layer
        top_heads = [(args.layer, head) for head in range(0,32)]
        heads = top_heads

        alphas = [args.alpha] * len(top_heads)

        best_alphas, metrics = optimize_alpha_for_layers(
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

        print(f"Best alphas for layer {layer}: {best_alphas}")

if __name__ == "__main__":
    main()   
    