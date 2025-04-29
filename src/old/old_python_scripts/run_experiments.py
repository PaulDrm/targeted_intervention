import argparse
import os
import pandas as pd
import torch
import numpy as np
from einops import rearrange
import json

def run_experiment(seed):
    parser = argparse.ArgumentParser()
    # Add all your argument definitions here
    parser.add_argument("--model_name", type=str, default='llama_7B', help='model name')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.5)
    parser.add_argument('--use_center_of_mass', type=lambda x: (str(x).lower() == 'true'), help='Whether to use the center of mass or not')
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument("--list_of_heads", type=str, default="", action=ParseListOfLists, help="Input should be a list of lists (e.g., [['11', '0'], ['13', '0']]).")
    parser.add_argument('--input_path', type=str, help='input path')
    parser.add_argument('--output_path', type=str, help='output path')
    parser.add_argument('--dataset_name', type=str, default='requirements_data')
    parser.add_argument('--add_or_subtract', type=lambda x: (str(x).lower() == 'true'), default='true', help='if intervention is added or substract to activations')
    parser.add_argument('--test_set_input_path', type=str)
    parser.add_argument('--prompt_type', type=str, default="open_ended")
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--consistency_factor', type=int, default=6)

    args = parser.parse_args()

    # Override the seed with the one passed to the function
    args.seed = seed

    if not os.path.exists(f"{args.output_path}"):
        os.mkdir(f"{args.output_path}")

    # Print all arguments
    print(f"Running experiment with seed: {seed}")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    if args.list_of_heads != "":
        list_of_heads = [[int(head[0]), int(head[1])] for head in args.list_of_heads]
        print("Parsed list of lists:", list_of_heads)
    else:
        list_of_heads = None

    df = pd.read_json(args.input_path)

    # Set the appropriate id_column based on the dataset
    id_column = "data_id" if args.dataset_name != "requirements_data" else "req_id"

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Process the data
    index_dic, separated_activations, separated_labels, reqs_order = process_data(df, id_column)

    # Create folds
    number_of_examples = np.arange(len(reqs_order))    
    fold_idxs = np.array_split(number_of_examples, args.num_fold)

    # Run the experiment for each fold
    for i in range(args.num_fold):
        run_fold(i, args, fold_idxs, reqs_order, index_dic, df, separated_activations, separated_labels)

def process_data(df, id_column):
    index_dic = {}
    separated_activations = []
    separated_labels = []
    reqs_order = []
    use_attention = True
    column = "attentions" if use_attention else 'o_proj_activations'
    
    for req_id in df[id_column].unique():
        req_df = df[df[id_column] == req_id].index
        index_dic[req_id] = list(req_df)
        
        temp_activations = df[df[id_column] == req_id][column]
        activations = np.array([list(sample.values()) for sample in temp_activations.values])
        batch_length = len(temp_activations)
        dim = 128
        activations = np.reshape(activations, (batch_length, 32, 32, dim))

        temp_labels = [1 if label else 0 for label in df[df[id_column] == req_id]['correct'].values]
        separated_labels.append(temp_labels)
        separated_activations.append(activations)
        reqs_order.append(req_id)

    return index_dic, separated_activations, separated_labels, reqs_order

def run_fold(fold_index, args, fold_idxs, reqs_order, index_dic, df, separated_activations, separated_labels):
    # Determine train, validation, and test sets
    train_idxs, val_set_idxs, test_idxs = get_fold_indices(fold_index, args, fold_idxs)

    # Get the top heads
    top_heads= args.list_of_heads

    print("Heads intervened: ", sorted(top_heads))
    print("Number of heads intervened: ", len(top_heads))
    top_heads = [[int(item) for item in tup] for tup in top_heads]
    print("Top heads: ", top_heads)

    # Get the interventions
    tuning_activations = np.concatenate(separated_activations, axis=0)
    com_directions = get_com_directions(32, 32, train_idxs, val_set_idxs, separated_activations, separated_labels) if args.use_center_of_mass else None
    interventions = get_interventions_dict(top_heads, probes, tuning_activations, 32, args.use_center_of_mass, args.use_random_dir, com_directions)

    # Run the intervention
    test_set = prepare_test_set(args)
    results = run_llama_intervention(args, tokenizer, model, interventions, test_set)

    # Process and save results
    process_results(results, args, top_heads, fold_index, seed)

def get_fold_indices(fold_index, args, fold_idxs, reqs_order):
    # if args.num_fold == 1:
    #     train_idxs = np.arange(len(reqs_order))
    # else:
    #     train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != fold_index])
    
    # rng = np.random.default_rng(args.seed)
    # size = int(len(train_idxs) * (1 - args.val_ratio))
    # train_set_idxs = rng.choice(train_idxs, size=size, replace=False)
    # val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

    # test_idxs = val_set_idxs if len(fold_idxs) == 1 else fold_idxs[fold_index]
        ## Set up random number generator
    rng = np.random.default_rng(args.seed)

        # Create indices for all samples
    all_indices = np.arange(len(reqs_order))

    # Shuffle indices
    rng.shuffle(all_indices)

    # Split indices into folds
    fold_size = len(all_indices) // args.num_fold
    fold_indices = [all_indices[i:i+fold_size] for i in range(0, len(all_indices), fold_size)]

    test_idxs = fold_indices[fold_index]
    train_idxs = np.concatenate([fold_indices[i] for i in range(args.num_fold) if i != fold_index])
    val_idxs = train_idxs
    
    return train_idxs, val_idxs, test_idxs

def prepare_test_set(test_set, args):
    
    #test_set = pd.read_json(args.test_set_input_path)
    test_set.reset_index(drop=True, inplace=True)
    indexes = [test_set[test_set['req_id'] == req_id].index[0] for req_id in test_set.req_id.unique()]
    repeated_indexes = indexes * args.consistency_factor
    return test_set.loc[repeated_indexes]

def process_results(results, args, top_heads, fold_index):
    curr_fold_results = pd.DataFrame(results)
    
    head_string = "_".join([f"{head[0]}_{head[1]}" for head in top_heads])
    
    curr_fold_results.to_json(f"{args.output_path}/results_intervention_{int(args.alpha)}_number_heads_{len(top_heads)}.json", orient='records', indent=4)
    
    if args.prompt_type != "open_ended":
        print(curr_fold_results.predict.value_counts())

        with open(f'{args.output_path}/overall_results.txt', 'a') as f:
            print(f"For dataset fold {fold_index} and alpha {args.alpha} and heads {head_string}", file=f)
            print(curr_fold_results.predict.value_counts(), file=f)
            print(curr_fold_results.final_answer.value_counts(), file=f)

        value_counts = curr_fold_results.predict.value_counts().to_dict()
        value_counts['alpha'] = args.alpha
        value_counts['heads'] = top_heads
        value_counts['seed'] = args.seed

        json_file_path = f'{args.output_path}/overall_results.json'
        
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []

        data.append(value_counts)

        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 1010]  # Add more seeds as needed

    # Load the model
    tokenizer, model = load_model(args.model_name, device="cuda:1")

    for seed in seeds:
        run_experiment(seed)
