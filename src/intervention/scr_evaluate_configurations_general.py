
import pandas as pd
import os

import argparse

import json

from utils.ut_processing_utils import get_activations_bau, load_model

from utils.ut_evaluation_utils import extract_answer_compare_gt, evaluate_configuration_general

from utils.ut_processing_utils import prepare_prompt, parse_output

import numpy as np

import ast 


def save_results(args, heads , alphas, fold_index, results, metrics):

    output_string = f"{args.output_path}/results_intervention_seed_{int(args.seed)}_alpha_{str(round(alphas[0],2)).replace('.', '_')}"

    results['heads'] = [heads for _ in range(len(results))]
    results['alphas'] = [heads for _ in range(len(results))]

    if not hasattr(args, 'layer'):            

        layer = heads[0][0]
        head_string = ""
        for head in heads:#list_of_heads:
            head_string = head_string + str(head[0]) + "_"+ str(head[1]) + "_"

        #curr_fold_results.to_json(f"{args.output_path}/results_test_intervention_seed_{int(args.seed)}_alpha_{str(round(alphas[0],2)).replace('.', '_')}_heads_{head_string}_fold_{fold_index}.json", orient='records', indent=4)
        output_string += f"_heads_{head_string}"
    else:
        
        if hasattr(args, 'head'):            
            #curr_fold_results.to_json(f"{args.output_path}/results_test_intervention_seed_{int(args.seed)}_alpha_{str(round(alphas[0],2)).replace('.', '_')}_head_{args.layer}_{args.head}_fold_{fold_index}.json", orient='records', indent=4)
            output_string += f"_head_{args.layer}_{args.head}"

        else: 
            #curr_fold_results.to_json(f"{args.output_path}/results_test_intervention_seed_{int(args.seed)}_alpha_{str(round(alphas[0],2)).replace('.', '_')}_layer_{args.layer}_fold_{fold_index}.json", orient='records', indent=4)

            output_string += f"_layer_{args.layer}"

    if hasattr(args, 'consistency_factor'):
        if args.consistency_factor!= 1:
            output_string += f"const_factor_{args.consistency_factor}"

    output_string += f"_fold_{fold_index}.json"
    try:
        results.to_json(output_string, orient='records', indent=4)
    except OSError as e: ## file name too long 
        print(f"Error occurred while writing to {output_string}: {e}")
        # Split the original file name if it's too long
        base_name, ext = os.path.splitext(output_string)
        split_point = len(base_name) // 2
        shortened_name = base_name[:split_point] + base_name[split_point + 1:]  # Simplify by truncating middle
        fallback_file_name = shortened_name[:50] + ext  # Ensure a reasonable length for the fallback name
        results.to_json(fallback_file_name, orient='records', indent=4)

    if args.prompt_type!= "open_ended":

        value_counts = results.predict.value_counts().to_dict()
        value_counts['alpha'] = alphas#args.alpha
        value_counts['layer'] = args.layer if hasattr(args, 'layer') else layer
        value_counts['seed'] = args.seed
        # value_counts['precision'] = precision
        # value_counts['recall'] = recall
        # value_counts['precision_consistency'] = precision_consist
        # value_counts['recall_consistency'] = recall_consist
        value_counts['metrics'] = metrics
        
        if "length_normalized_entropy" in results.columns: 
            value_counts['length_normalized_entropy'] = results.length_normalized_entropy.sum()

        #if args.head != None: 
        if hasattr(args, 'head'):
            value_counts['head'] = args.head
        
        elif heads: 
            value_counts['head'] = heads

        if hasattr(args, 'consistency_factor'):
            value_counts['consistency_factor'] = args.consistency_factor

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
    parser.add_argument("--consistency_factor", type=int, default=4)
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
    parser.add_argument("--temperature", type=float, default=0.3)
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
    
    tokenizer, model = load_model(args.model_name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Check if input_path_configuration is not None or "null"
    if args.input_path_configuration and args.input_path_configuration.lower() != "null":
        log_filename_input = args.input_path_configuration

        log_filename_complete = os.path.join(args.output_path, log_filename_input)

        configurations = pd.read_csv(log_filename_complete,
                comment='#',  # <- This tells pandas to ignore lines starting with #
                converters={
                'heads': ast.literal_eval,
                'alphas': ast.literal_eval,
                'seeds': ast.literal_eval
                }) 
        
        configurations = configurations.to_dict('records')

    else:
        #log_filename_input = "configuration_optimization.csv"
        configurations = [{'heads': ((0,0),), 'alphas': [0],'seeds': (1234,)}]

    args.seed = 1234 #args.seeds[0]
    args.max_new_tokens = 600
    args.num_heads = model.config.num_attention_heads

    dataset = pd.read_json(args.input_path)

    from utils.ut_processing_utils import get_attention_activations
    from utils.ut_processing_utils import get_activations_bau

    dataset =get_attention_activations(dataset, tokenizer, model, get_activations_bau)

    print(dataset.complete_inputs.values[0])

    external_data_set =pd.read_json(args.test_set)

    #external_data_set.drop_duplicates(subset=['question'], inplace=True)

    external_data_set = external_data_set.reset_index(drop=True)
    #external_data_set.drop(columns=['prompt'], inplace=True)
    #external_data_set.rename(columns={'question': 'prompt'}, inplace=True)

    external_data_set['prompt'] = external_data_set['question']

    print("Configurations: ", configurations)
    for configuration in configurations:
        print("Configuration: ", configuration)
        alphas = configuration["alphas"]
        heads = configuration["heads"]

        results = evaluate_configuration_general(args, tokenizer, model, heads, alphas, dataset, external_test_set =external_data_set)

        # if curr_fold_results.predict.value_counts().to_dict().get("undefined", 0) > curr_fold_results.shape[0]/3:
        #     print("Warning: Undefined is predicted for more than 1/3 of the data.")
        #     precision = 0 
        #     undefined = 1
        if args.prompt_type == "ab_cot":

            results = extract_answer_compare_gt(args, results)

            curr_fold_results = pd.DataFrame(results)


            if args.dataset_name == "requirements_data":

                # Calculate precision and recall
                precision, recall = get_precision_recall(curr_fold_results)
                precision_consist, recall_consist = precision_recall_consistency(curr_fold_results)

                print("Precision: ", precision," Recall: ", recall)
                print("Precision consistency: ", precision_consist," Recall consistency: ", recall_consist)

                # Store the results
                precision_scores.append({"precision": precision, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas, "precision_consistency": precision_consist })
                recall_scores.append({"recall": recall, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas, "recall_consistency": recall_consist})

                metrics = {"precision": precision, "recall": recall, "precision_consistency": precision_consist, "recall_consistency": recall_consist}

                save_results(args, heads , alphas, fold_index, results, metrics)
            
            else:

                fold_index = 0
                metrics = {"accuracy": curr_fold_results.correct.value_counts().to_dict().get(True, 0) / curr_fold_results.shape[0]}
                save_results(args, heads , alphas, fold_index,  curr_fold_results, metrics)

        else:
            
            curr_fold_results = pd.DataFrame(results)

            fold_index = 0
            metrics = {}
            save_results(args, heads , alphas, fold_index,  curr_fold_results, metrics)
  

if __name__ == "__main__":
    main()