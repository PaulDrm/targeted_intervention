
import torch
from tqdm import tqdm
import numpy as np

from einops import rearrange

from utils.ut_processing_utils import layer_head_to_flattened_idx, flattened_idx_to_layer_head, prepare_prompt, extract_final_answer_dataset, get_fold_indices

import sys 
sys.path.append('../')
from intervention.reasoning import eval_intervention, eval_intervention_batch, parse_output, evaluate, eval_intervention_batch_parallel


def get_top_heads(train_idxs, val_idxs, separated_activations, separated_labels, num_layers, num_heads, seed, num_to_intervene, use_random_dir=False, specific_heads = None):
    
    if specific_heads is not None:
        #top_heads = [layer_head_to_flattened_idx(head[0], head[1], num_heads) for head in specific_heads]
        probes, all_head_accs_np = train_probes(seed, train_idxs, val_idxs, separated_activations, separated_labels, num_layers=num_layers, num_heads=num_heads, heads=specific_heads)
        top_heads = specific_heads[:num_to_intervene]

    else:
        top_heads = []
        probes, all_head_accs_np = train_probes(seed, train_idxs, val_idxs, separated_activations, separated_labels, num_layers=num_layers, num_heads=num_heads)
    
        all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)
        top_accs = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene]
        #print(top_accs[0:5])
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]

    if use_random_dir: 
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(num_heads*num_layers, num_heads*num_layers, replace=False)
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

    return top_heads, probes

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def grid_search(X_train, y_train):

    # Define the grid of hyperparameters to search
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],#[0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
        'penalty': ['l2', 'l1'],
        'max_iter': [100, 1000, 10000]
    }

    # Define the model
    model = LogisticRegression()

    # Setup the grid search
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters and best score
    # print("Best Parameters:", grid_search.best_params_)
    # print("Best Score:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_score_


def train_probes(seed, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, heads =None):
    
    all_head_accs = []
    probes = []

    all_X_train = np.concatenate([separated_head_wise_activations[i] for i in train_set_idxs], axis = 0)
    print(all_X_train.shape)
    all_X_val = np.concatenate([separated_head_wise_activations[i] for i in val_set_idxs], axis = 0)
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis = 0)
    rm_outliers = False
    #heads= None
    if rm_outliers:
                print("outliers will be removed")
    if heads == None:

        for layer in tqdm(range(num_layers)):
            #print("Test") 

            #rm_outliers = False #True
            
            for head in range(num_heads): 
                X_train = all_X_train[:,layer,head,:]
                X_val = all_X_val[:,layer,head,:]
                
                #clf = LogisticRegression(random_state=seed, max_iter=10000).fit(X_train, y_train)
                #clf = LogisticRegression(random_state=seed, C=1000, penalty='elasticnet', l1_ratio= 0.5, max_iter=10000, solver='saga').fit(X_train, y_train) #, penalty='l1', , C=100000
                #clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X_train, y_train)
                # y_pred = clf.predict(X_train)
                # y_val_pred = clf.predict(X_val)
                # all_head_accs.append(accuracy_score(y_val, y_val_pred))
                # probes.append(clf)
                
                y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
                #print(y_train.shape)
                y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis = 0)
                # Filtering for outliers
                
                #if rm_outliers:
                    #print("Removing outliers")
                #    X_val, y_val = remove_outliers(X_train, X_val, y_val)
                #    X_train, y_train = remove_outliers(X_train, X_train, y_train)

                #clf, train_acc = grid_search(X_train, y_train)

                clf = LogisticRegression(random_state=seed, max_iter=10000).fit(X_train, y_train)

                y_val_pred = clf.predict(X_val)
                all_head_accs.append(accuracy_score(y_val, y_val_pred))

                probes.append(clf)

    else: 
        print(f"Running probes for {str(heads)} heads")
        reduce_dim = False
         #True
        #slice_id = 128
        #probes = np.zeros(1024)
        # Desired length of the list
        length = 1024
        # Create a list of a specific length filled with None (or any other placeholder)
        probes = [None] * length
        for h in tqdm(heads): 
            layer = h[0]
            head = h[1]

            X_train = all_X_train[:,layer,head,:]#slice_id]
            X_val = all_X_val[:,layer,head,:]#slice_id]
            
            y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
            #print(y_train.shape)
            y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis = 0)
            # Filtering for outliers

            #### CAREFUL PCA IS ON
            ## Get linear coefficients from reduced dimensional space 
            if reduce_dim: 
                dim_reduction = PCA(n_components = 3, random_state= 22).fit(X_train)
                X_train_transform = dim_reduction.transform(X_train)
                X_val_transform = dim_reduction.transform(X_val)
                X_train = X_train_transform
                X_val = X_val_transform

            #if rm_outliers:
            #    print("Removing outliers")
            #    X_val, y_val = remove_outliers(X_train, X_val, y_val)
            #    X_train, y_train = remove_outliers(X_train, X_train, y_train)

            clf, train_acc = grid_search(X_train, y_train)

            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))

            if reduce_dim: 
                clf.reduced_coef_ = clf.coef_
                clf.coef_ = dim_reduction.inverse_transform(clf.coef_)

            #clf.coef_ = dim_reduction.components_[1]
            # from sentence_transformers import util
            # print(util.cos_sim(clf.coef_[0],comparison))
            index = layer_head_to_flattened_idx(layer, head, 32)

            #probes.append(clf)
            probes[index] = clf

    all_head_accs_np = np.array(all_head_accs)
    return probes, all_head_accs_np


def lt_modulated_vector_add(head_output, layer_name, interventions = {}, args = None, start_edit_location='lt'):#, add_proj_val_std = args.add_proj_val_std ): 

        ## applies same alpha to all heads 

        head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=args.num_heads)
        for head, direction, proj_val_std in interventions[layer_name]:
            #print(head)
            #print(direction)
            #print(direction.dtype)
            direction_to_add = torch.tensor(direction).to(head_output.device.index)
            
            if start_edit_location == 'lt': 
                head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add  
                
            else: 
                head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
            #print(head_output[:, -1, head, :])

        head_output = rearrange(head_output, 'b s h d -> b s (h d)')
        return head_output

def lt_modulated_vector_no_alpha(head_output, layer_name, interventions = {}, args = None, start_edit_location='lt'): 
        
        ## alphas have already been applied during direction calculation
        head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=args.num_heads)

        for head, direction, proj_val_std in interventions[layer_name]:
            #print(layer_name)
            #print(head)
            #print(direction)
            direction_to_add = torch.tensor(direction).to(head_output.device.index)

            if start_edit_location == 'lt': 
                head_output[:, -1, head, :] += proj_val_std * direction_to_add
            else: 
                head_output[:, start_edit_location:, head, :] +=  proj_val_std * direction_to_add
        
        head_output = rearrange(head_output, 'b s h d -> b s (h d)')
        return head_output

def get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels): 

    com_directions = []

    for layer in range(num_layers): 
        for head in range(num_heads): 
            usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
            # Filter by unique indices
            usable_idxs = np.unique(usable_idxs)
            # print(usable_idxs.shape)
            usable_head_wise_activations = np.concatenate([separated_head_wise_activations[i][:,layer,head,:] for i in usable_idxs], axis=0)
            # print(usable_head_wise_activations.shape)
            usable_labels = np.concatenate([separated_labels[i] for i in usable_idxs], axis=0)
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions

def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir, com_directions): 

    interventions = {}
    use_attentions = True
    ## TODO This needs to be better visible
    if use_attentions: 
        module = "head_out"
    else: 
        module = "o_proj_out"

    for layer, head in top_heads: 
        #interventions[f"model.layers.{layer}.self_attn.head_out"] = []
        interventions[f"model.layers.{layer}.self_attn.{module}"] = [] 
    for layer, head in top_heads:
        if use_center_of_mass: 
            #print(f"Layer: {layer}, Head: {head}")
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir: 
            direction = np.random.normal(size=(128,))
        else: 
            direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        direction = direction / np.linalg.norm(direction)
        #print(direction.shape)
        activations = tuning_activations[:,layer,head,:direction.shape[-1]] # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.{module}"].append((head, direction.squeeze(), proj_val_std))
        
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.{module}"] = sorted(interventions[f"model.layers.{layer}.self_attn.{module}"], key = lambda x: x[0])

    return interventions

def get_interventions_dict_variable_alpha(top_heads, alphas, tuning_activations, num_heads, use_center_of_mass, use_random_dir, com_directions): 

    interventions = {}
    use_attentions = True
    ## TODO This needs to have a variable in the function
    if use_attentions: 
        module = "head_out"
    else: 
        module = "o_proj_out"

    module = "o_proj"

    for layer, head in top_heads: 
        #interventions[f"model.layers.{layer}.self_attn.head_out"] = []
        interventions[f"model.layers.{layer}.self_attn.{module}"] = [] 
    #for layer, head in top_heads:
    for h, alpha in zip(top_heads, alphas):
        layer, head = h
        layer = int(layer)
        head = int(head)
        if use_center_of_mass: 
            #print(f"Layer: {layer}, Head: {head}")
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir: 
            direction = np.random.normal(size=(128,))
        #else: 
        #    direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        
        ## TODO changed
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:,layer,head,:direction.shape[-1]] # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.{module}"].append((head, direction.squeeze(), alpha*proj_val_std))

        ### BETTER SCALING OF OPTIMIZED ALPHA? 
        #interventions[f"model.layers.{layer}.self_attn.{module}"].append((head, direction.squeeze(), alpha))
        
        #interventions[f"model.layers.{layer}.self_attn.{module}"].append((head, direction.squeeze(), proj_val_std))
        
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.{module}"] = sorted(interventions[f"model.layers.{layer}.self_attn.{module}"], key = lambda x: x[0])

    return interventions

import numpy as np

def get_interventions_dict_variable_alpha(
        top_heads,
        alphas,
        tuning_activations,
        num_heads,
        use_center_of_mass,
        use_random_dir,
        com_directions,
        *,
        module: str = "o_proj",
        key_template: str = "model.layers.{layer}.self_attn.{module}",
):
    """
    Build the interventions dictionary.

    `key_template` must include {layer}. It may also include {module}.
    """

    def key_for(layer_val: int) -> str:
        """Create the dict key for a given layer number."""
        if "{module}" in key_template:
            return key_template.format(layer=layer_val, module=module)
        return key_template.format(layer=layer_val)

    # Pre-create all keys so they appear in a stable order
    interventions = {}
    for layer, _ in top_heads:
        key = key_for(int(layer))
        if key not in interventions:          # avoid overwriting if repeated
            interventions[key] = []

    for (layer, head), alpha in zip(top_heads, alphas):
        layer = int(layer)
        head = int(head)
        key = key_for(layer)

        # Pick a direction vector
        if use_center_of_mass:
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir:
            direction = np.random.normal(size=(128,))
        else:
            raise ValueError("Set either use_center_of_mass or use_random_dir to True")

        direction = direction / np.linalg.norm(direction)

        # Project activations onto this direction
        activations = tuning_activations[:, layer, head, :direction.shape[-1]]
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)

        # Add entry to the dict
        if key not in interventions:
            interventions[key] = []
        interventions[key].append((head, direction.squeeze(), alpha * proj_val_std))

    # Sort heads inside each layer for stable order
    for key in interventions:
        interventions[key].sort(key=lambda x: x[0])

    return interventions


def get_interventions_dict_variable_alpha_no_norm(top_heads, alphas, tuning_activations, num_heads, use_center_of_mass, use_random_dir, com_directions): 

    interventions = {}
    use_attentions = True
    ## TODO This needs to have a variable in the function
    if use_attentions: 
        module = "head_out"
    else: 
        module = "o_proj_out"

    module = "o_proj"

    for layer, head in top_heads: 
        #interventions[f"model.layers.{layer}.self_attn.head_out"] = []
        interventions[f"model.layers.{layer}.self_attn.{module}"] = [] 
    #for layer, head in top_heads:
    for h, alpha in zip(top_heads, alphas):
        layer, head = h
        layer = int(layer)
        head = int(head)
        if use_center_of_mass: 
            #print(f"Layer: {layer}, Head: {head}")
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir: 
            direction = np.random.normal(size=(128,))
        #else: 
        #    direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        
        ## TODO changed
        # direction = direction / np.linalg.norm(direction)
        # activations = tuning_activations[:,layer,head,:direction.shape[-1]] # batch x 128
        # proj_vals = activations @ direction.T
        # proj_val_std = np.std(proj_vals)
        # interventions[f"model.layers.{layer}.self_attn.{module}"].append((head, direction.squeeze(), alpha*proj_val_std))

        ### BETTER SCALING OF OPTIMIZED ALPHA? 
        interventions[f"model.layers.{layer}.self_attn.{module}"].append((head, direction.squeeze(), alpha))
        
        #interventions[f"model.layers.{layer}.self_attn.{module}"].append((head, direction.squeeze(), proj_val_std))
        
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.{module}"] = sorted(interventions[f"model.layers.{layer}.self_attn.{module}"], key = lambda x: x[0])

    return interventions

def get_directions(model, df, args=None, id_column = "data_id", column = "activations"):

    fold_index = 0

    #### Purpose of this script is to process attention head activations from a dataframe
    #### So for every unique data_id, it separates the attention head activations into different batches and labels
    index_dic = {}

    ## List of lists with attention head activations for each data_id examples
    separated_activations = []

    ## List of lists with labels for each data_id examples
    separated_labels = []
    data_ids_order = []

    num_layers = model.config.num_hidden_layers 
    num_heads = model.config.num_attention_heads

    if hasattr(model.config, "head_dim"):
        head_dim = model.config.head_dim
    else:
        head_dim = model.config.hidden_size // num_heads

    for data_id in df[id_column].unique():

        ## Necessary? --> used later when expanding train idxs and used for verbose logging
        example_indexes = df[df[id_column] == data_id].index
        ## Gives indexes for samples in the whole dataset
        index_dic[data_id] = list(example_indexes)
        ## Example: {'304_a': [0, 2], '304_b': [1, 3], '294_a': [4, 6], '294_b': [5, 7]} --> Dataset with 4 unique ids with 2 examples each
        
        temp_activations = df[df[id_column] == data_id][column]
        
        activations = np.array([list(sample.values()) for sample in temp_activations.values]) # [num_examples, num_layers x num_heads, head_dim]
        
        ## Number of example for the current data_id
        number_examples = len(temp_activations)
        
        ## split into attention heads
        example_activations = np.reshape(activations, (number_examples, num_layers, num_heads, head_dim))
        example_labels =[1 if label==True else 0 for label in df[df[id_column] == data_id]['correct'].values]
        
        separated_activations.append(example_activations)
        separated_labels.append(example_labels)
        
        data_ids_order.append(data_id)

    train_set_idxs, val_set_idxs, test_idxs = get_fold_indices(fold_index, args, data_ids_order)
    com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_activations, separated_labels)


    return com_directions, separated_activations
