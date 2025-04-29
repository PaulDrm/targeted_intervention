

import pandas as pd 

import json
import torch 
import numpy as np

import os 

from utils.ut_intervention_utils import (get_com_directions,  
                                         get_interventions_dict_variable_alpha, 
                                         lt_modulated_vector_add, 
                                         lt_modulated_vector_no_alpha
)

from utils.ut_processing_utils import extract_final_answer_dataset

from utils.ut_run_llms import run_llama_intervention_batch, run_llama_intervention_batch_parallel

from utils.ut_processing_utils import process_data, prepare_test_set, get_fold_indices #, get_test_set_folds

from utils.ut_processing_utils import get_attention_activations

def get_precision_recall(df):

    df_predict = df
    #df_predict.loc[df_predict['predict'] == "undefined", 'predict'] = "no"
    #print(df_predict.columns)
    #print(df_predict)

    df_predict.loc[df_predict['final_answer'] == "undefined", 'final_answer'] = False

    predict_gpt4 = pd.read_json("../datasets/requirements_data/requirements_gt_1510.json")
    corrects = []
    for row in df_predict.iterrows(): 
        ground_truth = predict_gpt4[predict_gpt4['req_id'] == row[1]['req_id']]['gt'].item()
        
        ## False positives are direction we don't want --> negative label
        if row[1]['final_answer'] != ground_truth:# and row[1]['predict']== 'yes': 
            corrects.append(False)

        ## True positives and Negatives are desired --> positive label
        else: 
            corrects.append(True)

    df_predict['correct'] = corrects

    epsilon = 1e-7  # Small value to prevent division by zero

    df = df_predict
    true_positives = len(df[(df.final_answer == True) & (df.correct == True)])
    predicted_positives = len(df[df.final_answer == True])
    precision = true_positives / (predicted_positives +epsilon)

    true_positives = len(df[(df.final_answer == True) & (df.correct == True)])
    #false_negatives = len(df[(df.final_answer == False) & (df.correct == True)])
    false_negatives = len(df[(df.final_answer == False) & (df.correct == False)])
    
    recall = true_positives / (true_positives + false_negatives +epsilon)

    if precision == 0 and recall == 0:
        precision = 1

    return precision, recall

def generate_majority_predictions(df): 
    
    predict = []
    for req_id in df['req_id'].unique(): 

        req_df = df[df['req_id'] == req_id]
      
        maj_ele = req_df['final_answer'].value_counts().index[0]
        uncertainty = max(req_df['final_answer'].value_counts()) / len(req_df)
        mean_score = 0
        predict.append({"req_id": req_id, "majority_predict" : maj_ele, "uncertainty" : uncertainty, "mean_score": mean_score})

    predict = pd.DataFrame(predict)
    return predict

def precision_recall_consistency(df):

    maj_df = generate_majority_predictions(df)
    maj_df.rename(columns={"majority_predict": "predict"}, inplace=True)
    maj_df['final_answer'] = maj_df['predict']

    df_predict = maj_df
    df_predict.loc[df_predict['uncertainty'] <= 0.5, 'final_answer'] = False

    predict_gpt4 = pd.read_json("../datasets/requirements_data/requirements_gt_2701.json")
    
    corrects = []
    for row in df_predict.iterrows(): 
        ground_truth = predict_gpt4[predict_gpt4['req_id'] == row[1]['req_id']]['gt'].item()
        
        ## False positives are direction we don't want --> negative label
        if row[1]['predict'] != ground_truth: # and row[1]['predict']== 'yes': 
            corrects.append(False)
        ## True positives and Negatives are desired --> positive label
        else: 
            corrects.append(True)

    df_predict['correct'] = corrects

    epsilon = 1e-7  # Small value to prevent division by zero

    df = df_predict
    true_positives = len(df[(df.final_answer == True) & (df.correct == True)])
    predicted_positives = len(df[df.final_answer == True])
    precision = true_positives / (predicted_positives + epsilon)

    true_positives = len(df[(df.final_answer == True) & (df.correct == True)])
    
    #false_negatives = len(df[(df.final_answer == False) & (df.correct == True)])
    false_negatives = len(df[(df.final_answer == False) & (df.correct == False)])

    recall = true_positives / (true_positives + false_negatives + epsilon)


    if precision == 0 and recall == 0:
        precision = 1

    #print("Precision: ", precision, "Recall: ", recall)
    #print("Precision: ", precision)
    #print("Recall: ", recall)

    return precision, recall

def cos_sim_numpy(a, b):

    dot_product = np.dot(a, b)
    magnitude_1 = np.linalg.norm(a)
    magnitude_2 = np.linalg.norm(b)

    cosine_sim = dot_product / (magnitude_1 * magnitude_2)
    
    return cosine_sim

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


def evaluate_configuration_requirements(args, tokenizer, model, heads, alphas, external_test_set =None):
    
    # print(args.input_path)
    df = pd.read_json(args.input_path)
    
    if "attentions" not in df.columns:
        df = get_attention_activations(df, tokenizer, model)

        predict_gpt4 = pd.read_json("../datasets/requirements_data/requirements_gt_2701.json")

        correct = []
        for row in df.iterrows(): 
            ground_truth = predict_gpt4[predict_gpt4['req_id'] == row[1]['req_id']]['gt'].item()
            
            ## False positives are direction we don't want --> negative label
            if row[1]['final_answer'] != ground_truth and row[1]['predict']== 'yes': 
                correct.append(0)
            ## True positives and Negatives are desired --> positive label
            else: 
                correct.append(1)
        df['correct'] = correct
        
        #df['o_proj_activations'] = o_proj_activations_list
        df.to_json(args.input_path, orient="records", indent=4)

    # Set the appropriate id_column based on the dataset
    id_column = "data_id" if args.dataset_name != "requirements_data" else "req_id"
    num_heads =32
    precision_scores = []
    recall_scores = []

    ## This check if majority of prediction are "undefined", which gives a signal if the intervention are still well dimensioned
    undefined = 0 

    # Run for multiple seeds
    for seed in args.seeds:

        if undefined: 
            break

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

            layer = None
            #heads = None

            if undefined: 
                break

            # Determine train, validation, and test sets
            train_idxs, val_set_idxs, test_idxs = get_fold_indices(fold_index, args, reqs_order) #fold_idxs,
            print("Train indexes are: ", train_idxs)
            train_index_expanded = np.concatenate([list(index_dic.values())[i] for i in train_idxs])
            train_set = df.loc[train_index_expanded]
            print("Train labels are: ", train_set.correct.values)
            print("Shape of train set is :", len(train_idxs))
            # Get the top heads
            top_heads= heads# args.list_of_heads
            print(top_heads)

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

            if external_test_set is None:
                print("Test indexes are: ", test_idxs)
                test_index_expanded = np.concatenate([list(index_dic.values())[i] for i in test_idxs])
                # Run the intervention
                test_set = df.loc[test_index_expanded]
                #print("Test set shape is :", test_set.shape)
            
            else:
                test_set = external_test_set

            test_set = prepare_test_set(test_set, args)
            print("Test set shape is :", test_set.shape)
            test_set = test_set.iloc[0:int(test_set.shape[0])]# #6] # int(test_set.shape[0])]#int(test_set.shape[0])]
            #print(test_set.iloc[0:6])
            if test_set.shape[0] == 1:
                results = run_llama_intervention_batch(args, tokenizer, model, interventions, test_set)
            else:
                results = run_llama_intervention_batch_parallel(args, tokenizer, model, interventions, test_set)
            
            print(results)
            curr_fold_results = pd.DataFrame(results)

            # Calculate precision and recall
            #
            precision, recall = get_precision_recall(curr_fold_results)
            precision_consist, recall_consist = precision_recall_consistency(curr_fold_results)

            print("Precision: ", precision," Recall: ", recall)
            print("Precision consistency: ", precision_consist," Recall consistency: ", recall_consist)

            if curr_fold_results.predict.value_counts().to_dict().get("undefined", 0) > curr_fold_results.shape[0]/3:
                print("Warning: Undefined is predicted for more than 1/3 of the data.")
                precision = 0 
                undefined = 1

            # Store the results
            precision_scores.append({"precision": precision, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas, "precision_consistency": precision_consist })
            recall_scores.append({"recall": recall, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas, "recall_consistency": recall_consist})

            output_string = f"{args.output_path}/results_test_intervention_seed_{int(args.seed)}_alpha_{str(round(alphas[0],2)).replace('.', '_')}"

            if not hasattr(args, 'layer'):            

                layer = heads[0][0]
                head_string = ""
                for head in top_heads:#list_of_heads:
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
                curr_fold_results.to_json(output_string, orient='records', indent=4)
            except OSError as e: ## file name too long 
                print(f"Error occurred while writing to {output_string}: {e}")
                # Split the original file name if it's too long
                base_name, ext = os.path.splitext(output_string)
                split_point = len(base_name) // 2
                shortened_name = base_name[:split_point] + base_name[split_point + 1:]  # Simplify by truncating middle
                fallback_file_name = shortened_name[:50] + ext  # Ensure a reasonable length for the fallback name
                curr_fold_results.to_json(fallback_file_name, orient='records', indent=4)

            value_counts = curr_fold_results.predict.value_counts().to_dict()
            value_counts['alpha'] = alphas#args.alpha
            value_counts['layer'] = args.layer if hasattr(args, 'layer') else layer
            value_counts['seed'] = args.seed
            value_counts['precision'] = precision
            value_counts['recall'] = recall
            value_counts['precision_consistency'] = precision_consist
            value_counts['recall_consistency'] = recall_consist
            
            if "length_normalized_entropy" in curr_fold_results.columns: 
                value_counts['length_normalized_entropy'] = curr_fold_results.length_normalized_entropy.sum()

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

    return precision_scores, recall_scores, undefined

def extract_answer_compare_gt(args, dataset):

    for i, res in enumerate(dataset):

        if args.dataset_name != "requirements_data" and args.prompt_type != "open_ended":
            final_answer, predict = extract_final_answer_dataset(res['output'], cot=True, internal_cot=False, dataset=args.dataset_name)
            
            gt = res['gt'].strip()
            correct = gt == predict
            
            res['predict'] = predict
            res['correct'] = correct
    
    return dataset

def evaluate_configuration_general(args, tokenizer, model, heads, alphas, dataset, external_test_set =None):
    
    num_layers = model.config.num_hidden_layers 
    num_heads = model.config.num_attention_heads
    #head_dim = model.config.head_dim
    
    ## This check if majority of prediction are "undefined", which gives a signal if the intervention are still well dimensioned
    undefined = 0 

    seed = args.seed

    args.seed = seed
        # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Process the data
    index_dic, separated_activations, separated_labels, data_ids_order = process_data(dataset, model)

    # # Create folds
    # number_of_examples = np.arange(len(data_ids_order)) 
    
    # #print(args.num_fold)
    # fold_idxs = np.array_split(number_of_examples, args.num_fold)

    # Run the experiment for each fold

    overall_results = []

    for fold_index in range(args.num_fold):

        #layer = None
        #heads = None

        if undefined: 
            break

        # Determine train, validation, and test sets
        train_idxs, val_set_idxs, test_idxs = get_fold_indices(fold_index, args, data_ids_order) #fold_idxs,
        
        ## verbose logging to sanity check train_set 
        print("Train indexes are: ", train_idxs)
        train_index_expanded = np.concatenate([list(index_dic.values())[i] for i in train_idxs])
        train_set = dataset.loc[train_index_expanded]
        print("Train labels are: ", train_set.correct.values)
        print("Shape of train set is :", len(train_idxs))
        # Get the top heads
        top_heads= heads# args.list_of_heads
        print("Heads intervened: ", sorted(top_heads))            
        
        # Get the interventions
        tuning_activations = np.concatenate(separated_activations, axis=0)
        com_directions = get_com_directions(num_layers, num_heads, train_idxs, val_set_idxs, separated_activations, separated_labels) if args.use_center_of_mass else None
        
        print(f"Evaluating configuration for head {heads} with alphas {alphas} for seed {args.seed} and fold {fold_index}")
        
        # Prepare the interventions dictionary
        interventions = get_interventions_dict_variable_alpha(
            heads, alphas, tuning_activations, num_heads, args.use_center_of_mass,
            args.use_random_dir, com_directions
        )

        #print("Interventions: ", interventions)


        if external_test_set is None:
            print("Test indexes are: ", test_idxs)
            test_index_expanded = np.concatenate([list(index_dic.values())[i] for i in test_idxs])
            # Run the intervention
            test_set = dataset.loc[test_index_expanded]
            #print("Test set shape is :", test_set.shape)
        
        else:
            test_set = external_test_set

        test_set = prepare_test_set(test_set, args)
        print("Test set shape is :", test_set.shape)
        test_set = test_set.iloc[0:int(test_set.shape[0])]# #6] # int(test_set.shape[0])]#int(test_set.shape[0])]
        #print(test_set.iloc[0:6])
        if test_set.shape[0] == 1:
            results = run_llama_intervention_batch(args, tokenizer, model, interventions, test_set)
        else:
            results = run_llama_intervention_batch(args, tokenizer, model, interventions, lt_modulated_vector_no_alpha, test_set, batch_size=2)
        
        for result in results:
            overall_results.append(result)
    #print(results)
    
    return overall_results
        
def run_configuration_multi_seeds_precision_recall(args, tokenizer, model, dataset, external_test_set=None):

    undefined = 0 
    # Run for multiple seeds
    for seed in args.seeds:

        if undefined: 
            break
        
        results = evaluate_configuration_general(args, tokenizer, model, heads, alphas, dataset, external_test_set)

        curr_fold_results = pd.DataFrame(results)

        # Calculate precision and recall
        #
        precision, recall = get_precision_recall(curr_fold_results)
        precision_consist, recall_consist = precision_recall_consistency(curr_fold_results)

        print("Precision: ", precision," Recall: ", recall)
        print("Precision consistency: ", precision_consist," Recall consistency: ", recall_consist)

        if curr_fold_results.predict.value_counts().to_dict().get("undefined", 0) > curr_fold_results.shape[0]/3:
            print("Warning: Undefined is predicted for more than 1/3 of the data.")
            precision = 0 
            undefined = 1

        # Store the results
        precision_scores.append({"precision": precision, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas, "precision_consistency": precision_consist })
        recall_scores.append({"recall": recall, "fold": fold_index, "seed": seed, "heads": heads, "alphas": alphas, "recall_consistency": recall_consist})

        output_string = f"{args.output_path}/results_test_intervention_seed_{int(args.seed)}_alpha_{str(round(alphas[0],2)).replace('.', '_')}"

        if not hasattr(args, 'layer'):            

            layer = heads[0][0]
            head_string = ""
            for head in top_heads:#list_of_heads:
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
            curr_fold_results.to_json(output_string, orient='records', indent=4)
        except OSError as e: ## file name too long 
            print(f"Error occurred while writing to {output_string}: {e}")
            # Split the original file name if it's too long
            base_name, ext = os.path.splitext(output_string)
            split_point = len(base_name) // 2
            shortened_name = base_name[:split_point] + base_name[split_point + 1:]  # Simplify by truncating middle
            fallback_file_name = shortened_name[:50] + ext  # Ensure a reasonable length for the fallback name
            curr_fold_results.to_json(fallback_file_name, orient='records', indent=4)

        value_counts = curr_fold_results.predict.value_counts().to_dict()
        value_counts['alpha'] = alphas#args.alpha
        value_counts['layer'] = args.layer if hasattr(args, 'layer') else layer
        value_counts['seed'] = args.seed
        value_counts['precision'] = precision
        value_counts['recall'] = recall
        value_counts['precision_consistency'] = precision_consist
        value_counts['recall_consistency'] = recall_consist
        
        if "length_normalized_entropy" in curr_fold_results.columns: 
            value_counts['length_normalized_entropy'] = curr_fold_results.length_normalized_entropy.sum()

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

    return precision_scores, recall_scores, undefined


def append_to_log_file(args, filename, log_entry):
    # Convert log_entry to a DataFrame and append to CSV
    log_df = pd.DataFrame([log_entry])
    filename = f"{args.output_path}/{filename}"
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        log_df.to_csv(filename, mode="w", index=False)
    else:
        log_df.to_csv(filename, mode="a", index=False, header=False)