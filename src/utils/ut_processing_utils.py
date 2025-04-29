import subprocess
import numpy as np
import ast

from tqdm import tqdm

import argparse 
import torch

import sys 
sys.path.append('../')
from intervention.reasoning import extract_final_answer

import llama

import copy

import pandas as pd
import torch.nn as nn

from baukit.baukit import Trace, TraceDict

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

class ParseListOfLists(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            result = ast.literal_eval(values)
            if not all(isinstance(i, list) and len(i) == 2 for i in result):
                raise ValueError("Each sublist must contain exactly two elements.")
            setattr(namespace, self.dest, result)
        except ValueError as ve:
            raise argparse.ArgumentTypeError(f"Input error: {ve}")
        except:
            raise argparse.ArgumentTypeError("Input should be a list of lists with two strings each (e.g., [['11', '0'], ['13', '0']])")


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

    #print(f"Searching for best configuration for head {head}")
    # Filter entries for the specific head
    head_entries = [
        entry for entry in log 
        if entry['heads']==head
        #if len(entry["heads"]) == 1 and entry["heads"][0] == head
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

def get_best_head_metrics_seeds(log, head):
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

    #print(f"Searching for best configuration for head {head}")
    # Filter entries for the specific head
    head_entries = [
        entry for entry in log 
        if entry['heads']==head
        #if len(entry["heads"]) == 1 and entry["heads"][0] == head
    ]
    
    if not head_entries:
        return None
        
    # Sort by our criteria
    best_entry = max(
        head_entries,
        key=lambda x: (
            len(x["seeds"]),  # Prefer entries with more seeds
            x["precision"],
            x["recall"],
            -x["alphas"][0]  # Negative so lower alpha is preferred
        )
    )
    
    print(f"Best configuration for head {head}: {best_entry}")
    return best_entry




def get_llama_activations_bau(model, prompt, device): 

    #HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    ## https://github.com/likenneth/honest_llama/issues/7
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    O_PROJ = [f"model.layers.{i}.self_attn.o_proj_out" for i in range(model.config.num_hidden_layers)]
    
    #MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    #print(HEADS)

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+O_PROJ, retain_input=True) as ret:
           #output = model.generate(prompt, return_dict_in_generate=True, output_hidden_states = True, output_scores=True)
            output = model(prompt, output_hidden_states = True)#,return_dict_in_generate=True, output_scores=True)
        
            #output = output[1]
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            #head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            # change from ret[head].output  to ret[head].input

            ## get attention inputs before merging transformation layer ,
            ## https://github.com/likenneth/honest_llama/issues/7
            head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            #mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            #mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

            o_proj_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
            o_proj_hidden_states = torch.stack(o_proj_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, o_proj_hidden_states# mlp_wise_hidden_states, output

def get_gpu_memory():
    # This command returns GPU details including memory usage
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'],
                            capture_output=True, text=True)
    return result.stdout

def select_device(min_vram_gb):
    gpus = []
    # Convert GB to bytes (1 GB = 2**30 bytes)
    min_vram_bytes = min_vram_gb *(1024)# (2**30)

    allocated_memory = get_gpu_memory().strip().split("\n")
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)
        props = torch.cuda.get_device_properties(i)
        allocated_memory_device = int(allocated_memory[i].split(",")[1].strip())
        print(allocated_memory_device)
        free_vram = props.total_memory / (1024 ** 2) - allocated_memory_device#torch.cuda.memory_allocated(device=i)
        print(f"Device cuda:{i} has {free_vram:.2f} GB of free VRAM.")

        if free_vram > min_vram_bytes:
            gpus.append(f"cuda:{i}")
            #return f"cuda:{i}"

    if len(gpus) == 0:
        print("No GPU available with sufficient VRAM.")
        return None
    

    elif len(gpus) == 1:
        #print(f"Selected device: {gpus[0]}")
        return gpus[0]
    

    elif len(gpus) > 1:
        return gpus

def load_model(model_name, device="cuda:0"):

    print("Loading right tokenizer!")

    LOAD_8BIT = False #True
    BASE_MODEL = model_name

    #tokenizer = llama.LlamaTokenizer.from_pretrained(BASE_MODEL)
    multi_check = False
    if type(device) == list:
        multi_check =True
        num_gpus = len(device)
        device = torch.device("cuda:0")

    #tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    model = llama.LlamaForCausalLM.from_pretrained(BASE_MODEL, low_cpu_mem_usage=True, 
                                                   load_in_8bit=LOAD_8BIT,
                                                   torch_dtype=torch.bfloat16,
                                                   device_map=device)
    
    # ── Detect which Llama generation we have ──────────────────────────────
    is_llama3 = (
        tokenizer.vocab_size > 100_000                # 128 k vocab
        or tokenizer.bos_token.startswith("<|begin")  # <|begin_of_text|>
    )
    is_llama2 = tokenizer.bos_token == "<s>"          # classic Llama BOS


    # ── Apply the right padding / EOS settings ─────────────────────────────
    if is_llama3:
        LLAMA3_PAD_EOS_ID = 128_009                  # <|eot_id|>
        model.generation_config.pad_token_id = LLAMA3_PAD_EOS_ID
        tokenizer.eos_token_id = LLAMA3_PAD_EOS_ID
        tokenizer.pad_token    = tokenizer.eos_token

    elif is_llama2:
        # For Llama-2 chat checkpoints the safest pad is the unknown token
        tokenizer.pad_token = tokenizer.unk_token     # "<unk>"

    # if "Meta-Llama-3-8B-Instruct" in model.config._name_or_path:
    
    
    #     model.generation_config.pad_token_id = 128009#32007
        
    #     tokenizer.eos_token_id = 128009
    #     tokenizer.pad_token = tokenizer.eos_token
    
    # elif "Llama-2-7b-chat-hf" in model.config._name_or_path:
        
    #     tokenizer.pad_token = "<unk>"
        #chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + ' [INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        #tokenizer.chat_template = chat_template

    # if multi_check:
    #     print(f"Using {num_gpus} GPUs with DataParallel")
    #     model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    return tokenizer, model 

def load_model_peft(model_name, peft_model_path="", device="cuda:0"):
    print("Loading tokenizer and base model...")
    
    # Configuration flags
    LOAD_8BIT = False  # Set to True if you want to load the model in 8-bit precision
    BASE_MODEL = model_name
    
    print("Loading right tokenizer!")
    print(BASE_MODEL)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load the base model with the specified dtype
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16 if not LOAD_8BIT else torch.float16
    )
    
    # Load the PEFT configuration
    config = PeftConfig.from_pretrained(peft_model_path)
    
    # Load the PEFT model on top of the base model
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    # Move the model to the specified device
    peft_model.to(device)
    
    # **Merge PEFT weights with the base model**
    print("Merging PEFT weights with the base model...")
    model = peft_model.merge_and_unload()
    model.to(device)  # Ensure the merged model is on the correct device
    
    # Optionally, you can save the merged model for future use
    # merged_model.save_pretrained("path_to_save_merged_model")
    # tokenizer.save_pretrained("path_to_save_merged_model")
    
    print("Model and tokenizer loaded and merged successfully.")

    if "Meta-Llama-3-8B-Instruct" in model.config._name_or_path:
        model.generation_config.pad_token_id = 128009#32007
        
        tokenizer.eos_token_id = 128009
        tokenizer.pad_token = tokenizer.eos_token
    
    elif "Llama-2-7b-chat-hf" in model.config._name_or_path:

        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def prepare_prompt(prompt, tokenizer, dataset_name=None, verbose= False):
    if dataset_name == "requirements_data":
        chat_dict = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Let's analyze the given System and requirement step-by-step: "}
        ]
        
        if verbose:
            print(chat_dict)
        prompt = tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=False)
        prompt = prompt[:-len(tokenizer.eos_token)]
    # else:
    #     chat_dict = [
    #         {"role": "user", "content": prompt},
    #         {"role": "assistant", "content": "Let's analyze the problem step-by-step:"}
    #     ]
    #     #prompt = tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=True)
    #     prompt = tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=False, continue_final_message=True)

    else:
        chat_dict = [
            {"role": "user", "content": prompt}
        #    {"role": "assistant", "content": "Let's analyze the problem step-by-step:"}
        ]
        if verbose:
            print(chat_dict)
        prompt = tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=True)
        #prompt = tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=False, continue_final_message=True)


    return prompt

def generate_input_output_prompt(row, tokenizer):
    
    chat_dict = [
        {"role": "user", "content": row['prompt']},
        {"role": "assistant", "content": row['output']}]
    
    prompt = tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=False)#,tokenizer_kwargs={"add_special_tokens":False})
    
    return prompt

def parse_output(output, prompt, tokenizer):


    #print("Function parse_output: ", output)

    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    encoded_output = tokenizer.encode(output, add_special_tokens=False)

    #print("PAD Token ID: ", tokenizer.pad_token_id)
    encoded_output = [i for i in encoded_output if i!= tokenizer.pad_token_id]
    
    #print(encoded_prompt)
    is_llama2 = tokenizer.bos_token == "<s>"          # classic Llama BOS

    if is_llama2:
        ## This is neccesary because tokenizer.chat_template doesn't add a line space between <s> [INST], which is not the case in the normal tokenizer
         encoded_prompt += [0]

    # if tokenizer.name_or_path == 'meta-llama/Llama-2-7b-chat-hf':
    
    encoded_output = encoded_output[len(encoded_prompt):]
    output = tokenizer.decode(encoded_output, skip_special_tokens=False)
    #output =output[len(tokenizer.decode(encoded_prompt, skip_special_tokens=False)):]
    #print("Function parse_output: ", output)

    return output

def extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset='requirements_data'):

    if dataset== 'requirements_data':
        final_answer, predict = extract_final_answer(output, cot=cot, internal_cot=internal_cot)
    
    elif dataset == "golden_gate":

        if any([s in output.lower() for s in ["golden gate"]]):

            final_answer ="(A)"
            predict = "(A)"
        else:
            final_answer = "(B)"
            predict = "(B)"
    
    elif dataset == "truthfulqa_neon":

        if any([s in output.lower() for s in ["neon"]]):

            final_answer ="(A)"
            predict = "(A)"
        else:
            final_answer = "(B)"
            predict = "(B)"

    else: 
        matched_text = output
        #print(matched_text)

        if "(a)" in matched_text.lower() and "(b)" in matched_text.lower():
            final_answer = "undefined"

        elif "(a)" in matched_text.lower():
            final_answer = "(A)"
            
        elif "(b)" in matched_text.lower():
            final_answer = "(B)"
        else:
            final_answer = "undefined"  

        if final_answer == "(A)": 
            predict = "(A)" 

        elif final_answer == "undefined":
            predict = "undefined"   
        else:
            predict = "(B)"

    return final_answer, predict

def process_data(df, model, id_column="data_id"):
    
    #### Purpose of this script is to process attention head activations from a dataframe
    #### So for every unique data_id, it separates the attention head activations into different batches and labels

    ## List of lists with attention head activations for each data_id examples
    separated_activations = []

    ## List of lists with labels for each data_id examples
    separated_labels = []
    data_ids_order = []
    ## dataset column with activations dict shape num_layer x num_heads
    column = "activations" 
    index_dic = {}

    num_layers = model.config.num_hidden_layers 
    num_heads = model.config.num_attention_heads
    
    
    if hasattr(model.config, "head_dim"):
        head_dim = model.config.head_dim
    else:
        head_dim = model.config.hidden_size // num_heads

    for data_id in df[id_column].unique():

        ## Necessary? not for now at least
        example_indexes = df[df[id_column] == data_id].index
        ## Gives indexes for samples in the whole dataset
        index_dic[data_id] = list(example_indexes)
        
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

    return index_dic, separated_activations, separated_labels, data_ids_order

def get_llama_activations_bau(model, prompt, device): 

    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    ## https://github.com/likenneth/honest_llama/issues/7
    #HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    O_PROJ = [f"model.layers.{i}.self_attn.o_proj_out" for i in range(model.config.num_hidden_layers)]
    
    #MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    #print(HEADS)

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+O_PROJ, retain_input=True) as ret:
           #output = model.generate(prompt, return_dict_in_generate=True, output_hidden_states = True, output_scores=True)
            output = model(prompt, output_hidden_states = True)#,return_dict_in_generate=True, output_scores=True)
        
            #output = output[1]
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu()#.numpy()
            hidden_states = hidden_states.to(torch.float16)
            hidden_states = hidden_states.numpy()
            #head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            # change from ret[head].output  to ret[head].input

            ## get attention inputs before merging transformation layer ,
            ## https://github.com/likenneth/honest_llama/issues/7
            head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
            
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze()#.numpy()
            
            head_wise_hidden_states_fl16 = head_wise_hidden_states.to(torch.float16)
            head_wise_hidden_states_fl16 = head_wise_hidden_states_fl16.numpy()
            #mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            #mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

            o_proj_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
            o_proj_hidden_states = torch.stack(o_proj_hidden_states, dim = 0).squeeze()#.numpy()

    return hidden_states, head_wise_hidden_states_fl16, o_proj_hidden_states# mlp_wise_hidden_states, output


def get_fold_indices(fold_index, args, data_id_order, verbose=True):

    """
    Splits data into training, validation, and test indices based on the number of folds and validation ratio.

    Parameters:
    - reqs_order (list or array): The dataset to be split.
    - num_fold (int): Number of folds for cross-validation.
    - seed (int): Random seed for reproducibility.
    - val_ratio (float, optional): Ratio of the dataset to be used as validation set. Defaults to 0.0.
    - fold_index (int, optional): Index of the fold to be used as validation set. Defaults to 0.

    Returns:
    - train_idxs (array): Training indices.
    - val_idxs (array): Validation indices (currently same as train_idxs).
    - test_idxs (array): Validation set indices based on val_ratio or fold.
    """

    seed = args.seed
    val_ratio = args.val_ratio
    num_fold = args.num_fold

    assert num_fold > 0, "num_fold must be greater than 0."
    assert 0.0 <= val_ratio <= 1.0, "val_ratio must be in the range [0.0, 1.0)."

    rng = np.random.default_rng(seed)

    # Create indices for how many samples there are
    all_indices = np.arange(len(data_id_order))

    # Shuffle indices
    rng.shuffle(all_indices)

    if num_fold < 1:
        raise ValueError("num_fold must be at least 1.")

    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError("val_ratio must be in the range [0.0, 1.0).")

    if num_fold == 1:
        if val_ratio > 0.0 and val_ratio < 1.0:
            split_point = int(len(all_indices) * (1 - val_ratio))
            train_idxs = all_indices[:split_point]
            test_idxs = all_indices[split_point:]
            val_idxs = train_idxs  # Preserve original behavior
            if verbose:
                print(f"Num_fold=1 with val_ratio={val_ratio}:")
                print(f"Training set size: {len(train_idxs)}")
                print(f"Validation set size: {len(test_idxs)}")
        elif val_ratio ==0:
            train_idxs = all_indices
            val_idxs = all_indices
            test_idxs = all_indices
            if verbose:
                print("Num_fold=1 without val_ratio:")
                print("All indexes are used for training and validation.")
                print(f"All sets size: {len(all_indices)}")
    else:
        fold_size = len(all_indices) // num_fold
        remainder = len(all_indices) % num_fold

        # Create fold indices
        fold_indices = []
        start = 0
        for i in range(num_fold):
            end = start + fold_size
            if i == num_fold - 1:
                end += remainder  # Add the remainder to the last fold
            fold = all_indices[start:end]
            fold_indices.append(fold)
            start = end

        if not (0 <= fold_index < num_fold):
            raise ValueError(f"fold_index must be in the range [0, {num_fold - 1}].")

        # Select the fold for validation
        test_idxs = fold_indices[fold_index]
        # Combine the rest for training
        train_idxs = np.concatenate([fold for idx, fold in enumerate(fold_indices) if idx != fold_index])

        if val_ratio > 0.0:
            split_point = int(len(train_idxs) * (1 - val_ratio))
            new_train_idxs = train_idxs[:split_point]
            new_test_idxs = train_idxs[split_point:]
            test_idxs = new_test_idxs  # Override with validation split
            val_idxs = new_train_idxs  # Preserve original behavior
            if verbose:
                print(f"Num_fold={num_fold} with fold_index={fold_index} and val_ratio={val_ratio}:")
                print(f"Training set size: {len(new_train_idxs)}")
                print(f"Validation set size: {len(test_idxs)}")
        else:
            val_idxs = train_idxs
            if verbose:
                print(f"Num_fold={num_fold} with fold_index={fold_index} without val_ratio:")
                print(f"Training set size: {len(train_idxs)}")
                print(f"Validation set size (same as training): {len(test_idxs)}")

    return train_idxs, val_idxs, test_idxs

def prepare_test_set(test_set, args):
    #test_set = pd.read_json(args.test_set_input_path)
    print(f"Repeating test set based on consistency factor {args.consistency_factor} times")
    test_set.reset_index(drop=True, inplace=True)
    indexes = [test_set[test_set['data_id'] == data_id].index[0] for data_id in test_set.data_id.unique()]
    repeated_indexes = indexes * args.consistency_factor
    return test_set.loc[repeated_indexes]

def get_activations_bau(model, prompt, device): 

    #HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    ## https://github.com/likenneth/honest_llama/issues/7
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    #O_PROJ = [f"model.layers.{i}.self_attn.o_proj_out" for i in range(model.config.num_hidden_layers)]
    
    #MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    #print(HEADS)

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS, retain_input=True) as ret:
           #output = model.generate(prompt, return_dict_in_generate=True, output_hidden_states = True, output_scores=True)
            output = model(prompt, output_hidden_states = True)#,return_dict_in_generate=True, output_scores=True)
        
            #output = output[1]
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu()#.numpy()
            #hidden_states = hidden_states.to(torch.float16)
            hidden_states = hidden_states.to(torch.float32)
            hidden_states = hidden_states.numpy()
            #head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            # change from ret[head].output  to ret[head].input

            ## get attention inputs before merging transformation layer ,
            ## https://github.com/likenneth/honest_llama/issues/7
            head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
            
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze()#.numpy()
            
            #head_wise_hidden_states_fl16 = head_wise_hidden_states.to(torch.float16)
            head_wise_hidden_states_fl16 = head_wise_hidden_states.to(torch.float32)
            head_wise_hidden_states_fl16 = head_wise_hidden_states_fl16.numpy()
            #mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            #mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

            #o_proj_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
            #o_proj_hidden_states = torch.stack(o_proj_hidden_states, dim = 0).squeeze()#.numpy()
    
    return hidden_states, head_wise_hidden_states_fl16 #,o_proj_hidden_states# mlp_wise_hidden_states, output


def get_attention_activations(df, tokenizer, model, function = get_activations_bau):#, device, args):

    try:
        df = df[df['predict']!= "too long"]
    
    except:
        pass

    if 'complete_inputs' not in df.columns:
        
        prompts = df.apply(lambda row: generate_input_output_prompt(row, tokenizer), axis=1)
        df['complete_inputs'] = prompts
    else: 

        prompts = df['complete_inputs']
    h = df['complete_inputs']
    attentions = []
    o_proj_activations_list = []
    
    device = "cuda:0"

    for prompt in tqdm(prompts):
        temp = {}
        #print(prompt)
        prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)['input_ids']
        
        layer_wise_activations, head_wise_activations  = function(model, prompt, device)

        ## transform output in original representation after attention head 
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        sequence_length = len(prompt[0])
        #print("Printing from get_attention_activations function")
        #print(sequence_length)
        
        if hasattr(model.config, "head_dim"):
            dimension_of_heads = model.config.head_dim
        else:
            dimension_of_heads = model.config.hidden_size // num_heads


        #print(head_wise_activations.shape) ## [Layers, sequence_length, hidden_size]
        head_wise_activations = head_wise_activations.reshape(num_layers, sequence_length, num_heads, dimension_of_heads)
        #print(head_wise_activations.shape) ## [Layers, sequence_length, number_of_heads, dimension_of_heads]
        for layer in range(num_layers):
            for head in range(num_heads):
                ## get representation of last token at layer [layer] and head [head] 
                temp[f"layer_{layer}_head_{head}"] = head_wise_activations[layer, -1, head, :]

        #print(temp)
        attentions.append(copy.deepcopy(temp))
    
    #print(attentions)
    df['activations'] = attentions

    # predict_gpt4 = pd.read_json("../datasets/requirements_data/requirements_gt_2701.json")

    # correct = []
    # for row in df.iterrows(): 
    #     ground_truth = predict_gpt4[predict_gpt4['req_id'] == row[1]['req_id']]['gt'].item()
        
    #     ## False positives are direction we don't want --> negative label
    #     if row[1]['final_answer'] != ground_truth and row[1]['predict']== 'yes': 
    #         correct.append(0)
    #     ## True positives and Negatives are desired --> positive label
    #     else: 
    #         correct.append(1)

    # df['correct'] = correct
    #df['o_proj_activations'] = o_proj_activations_list
    return df