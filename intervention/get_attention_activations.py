## Test reasoning

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer #, OPTForCausalLM, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import GenerationConfig

import os
import torch
import re
import copy
import argparse
import json
from tqdm import tqdm
import pandas as pd 
from baukit import Trace, TraceDict
import time
import numpy as np 
import openai
import copy
#from reasoning import load_model 
from honest_llama import llama

def load_model(model_name, load_in_8bit=False):

    LOAD_8BIT = load_in_8bit #True
    BASE_MODEL = model_name

    tokenizer = llama.LlamaTokenizer.from_pretrained(BASE_MODEL)
    model = llama.LlamaForCausalLM.from_pretrained(BASE_MODEL, low_cpu_mem_usage=True, load_in_8bit=LOAD_8BIT, torch_dtype=torch.float16, device_map="auto")

    if "openchat" in model.config._name_or_path:

        model.generation_config.pad_token_id = 0

    return tokenizer, model 

from baukit import Trace, TraceDict
def get_llama_activations_bau(model, prompt, device): 

    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    ## https://github.com/likenneth/honest_llama/issues/7
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
            hidden_states = hidden_states.detach().cpu().numpy()
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            
            ## get attention inputs before merging transformation layer ,
            ## https://github.com/likenneth/honest_llama/issues/7
            ## change from ret[head].output  to ret[head].input
            #head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            #mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            #mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states#, mlp_wise_hidden_states, output

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

def main(inputs):

    script_dir = os.path.abspath(os.path.dirname(__file__))
    #with open(os.path.join(script_dir, inputs["input_path"]), "r") as f:
    #    requirements = json.load(f)
    df = pd.read_json(os.path.join(script_dir, inputs["input_path"]))
    tokenizer, model = load_model(inputs['model'], load_in_8bit=False)
    device ="cuda"

    prompts = df['complete_inputs']
    attentions = []
    o_proj_activations_list = []
    for prompt in tqdm(prompts):
        temp = {}
        prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)['input_ids']
        layer_wise_activations, head_wise_activations, o_proj_activations = get_llama_activations_bau(model, prompt, device)

        ## transform output in original representation after attention head 
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        sequence_length = len(prompt[0])
        print(sequence_length)
        dimension_of_heads = model.config.hidden_size // num_heads
        print(head_wise_activations.shape) ## [Layers, sequence_length, hidden_size]
        head_wise_activations = head_wise_activations.reshape(num_layers, sequence_length, num_heads, dimension_of_heads)
        print(head_wise_activations.shape) ## [Layers, sequence_length, number_of_heads, dimension_of_heads]
        for layer in range(num_layers):
            for head in range(num_heads):
                ## get representation of last token at layer [layer] and head [head] 
                temp[f"layer_{layer}_head_{head}"] = head_wise_activations[layer, -1, head, :]


        attentions.append(copy.deepcopy(temp))
        # ## transform output in original representation after attention head 
        # num_layers = model.config.num_hidden_layers
        # num_heads = model.config.num_attention_heads
        # sequence_length = len(prompt[0])
        # print(sequence_length)
        # dimension_of_heads = model.config.hidden_size // num_heads
        # print(o_proj_activations.shape) ## [Layers, sequence_length, hidden_size]
        # o_proj_activations = o_proj_activations.reshape(num_layers, sequence_length, num_heads, dimension_of_heads)
        # print(o_proj_activations.shape) ## [Layers, sequence_length, number_of_heads, dimension_of_heads]
        # for layer in range(num_layers):
        #     for head in range(num_heads):
        #         ## get representation of last token at layer [layer] and head [head] 
        #         temp[f"layer_{layer}_head_{head}"] = o_proj_activations[layer, -1, head, :]

        # o_proj_activations_list.append(copy.deepcopy(temp))
    
    #print(attentions)
    df['attentions'] = attentions
    #df['o_proj_activations'] = o_proj_activations_list

    df.to_json(os.path.join(script_dir, inputs["output_path"]),orient="records", indent=4)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process the config file path.')
    parser.add_argument('config_path', help='Path to the config.json file')
    
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config= json.load(file)

    inputs = config 
    main(inputs)