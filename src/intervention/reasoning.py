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

import time

import numpy as np 
#import openai

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [],tokenizer= None, encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False

def evaluate(
    prompt,
    model = None,
    tokenizer = None,
    stopping_criteria = None,
    device = 'cuda',
    **kwargs,
    ):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        **kwargs,
    )
    with torch.no_grad():
        #with torch.cuda.amp.autocast():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
                output_hidden_states= True,
                output_scores=True,
                #output_attentions=True,
            )
            s = generation_output.sequences[0]
            
    output = tokenizer.decode(s)
    #print(output)

    return output, generation_output


def evaluate_batch(
    prompts,
    model=None,
    tokenizer=None,
    stopping_criteria=None,
    device='cuda',
    **kwargs,
):
    
    # Determine the device by accessing the first parameter
    try:
        device = next(model.parameters()).device
    except StopIteration:
        raise ValueError("The model has no parameters. Ensure the model is properly initialized.")
    
    #print(device)
    #model = model.module
    
    # Tokenize all prompts in the batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,  add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generation_config = GenerationConfig(**kwargs)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria,
            #output_hidden_states=True,
            output_scores=True,
        )

    # Decode all sequences in the batch
    outputs = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)

    return outputs, generation_output

def create_prompt_string(prompt_reason="", user_prompt="", definition = "", req= "", system_prompt = "", cot= True, internal_cot = False):
    if cot:
        return f"{user_prompt} System: {definition}\nRequirement: {req}\n{prompt_reason}\n{system_prompt} " +"Let's analyze the given System and requirement step-by-step:"
    elif internal_cot: 
        return  f"{user_prompt} System: {definition}\nRequirement: {req}\n{prompt_reason}\n "
    else: 
        return f"{user_prompt} System: {definition}\nRequirement: {req}\n{prompt_reason}\n{system_prompt} "


#from baukit import Trace, TraceDict
from baukit.baukit import Trace, TraceDict

from functools import partial

def eval_intervention(
    prompt,
    model = None,
    tokenizer = None,
    stopping_criteria = None,
    device = 'cuda',
    interventions={}, 
    intervention_fn=None,
    **kwargs,
    ):

    # device = model.device

    # Determine the device by accessing the first parameter
    try:
        device = next(model.parameters()).device
    except StopIteration:
        raise ValueError("The model has no parameters. Ensure the model is properly initialized.")
    
    print(device)
    model = model.module
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        **kwargs,
    )

     # --- intervention code --- #
    def id(head_output, layer_name): 
        return head_output

    if interventions == {}: 
        intervene = id
        layers_to_intervene = []
    else: 
        intervene = partial(intervention_fn, start_edit_location='lt')
        layers_to_intervene = list(interventions.keys())
    # --- intervention code --- #

    with torch.no_grad():

        with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
            #with torch.cuda.amp.autocast():
        
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    stopping_criteria=stopping_criteria,
                    output_hidden_states= True,
                    output_scores=True,
                    #output_attentions=True,
                )
                s = generation_output.sequences[0]

    output = tokenizer.decode(s, skip_special_tokens=False)
    #print(output
    return output, generation_output

def eval_intervention_batch(
    prompts,
    model=None,
    tokenizer=None,
    stopping_criteria=None,
    device='cuda',
    interventions={},
    intervention_fn=None,
    **kwargs,
):
    
    # Determine the device by accessing the first parameter
    try:
        device = next(model.parameters()).device
    except StopIteration:
        raise ValueError("The model has no parameters. Ensure the model is properly initialized.")
    
    #print(device)
    #model = model.module
    
    # Tokenize all prompts in the batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,  add_special_tokens=False)
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    #print(attention_mask)

    generation_config = GenerationConfig(**kwargs)

    # --- intervention code --- #
    def id(head_output, layer_name):
        return head_output

    if interventions == {}:
        intervene = id
        layers_to_intervene = []
    else:
        intervene = partial(intervention_fn, start_edit_location='lt')
        layers_to_intervene = list(interventions.keys())
    # --- intervention code --- #

    with torch.no_grad():
        with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
                output_hidden_states=False,
                output_scores=True,
            )
    
    # Decode all sequences in the batch
    outputs = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)

    return outputs, generation_output

def eval_intervention_batch_parallel(
    prompts,
    model=None,
    tokenizer=None,
    stopping_criteria=None,
    device='cuda',
    interventions={},
    intervention_fn=None,
    **kwargs,
):
    

    # Split the prompts into sub-batches manually
    num_gpus = torch.cuda.device_count()
    split_size = (len(prompts) + num_gpus - 1) // num_gpus  # Calculate size of each split
    sub_batches = [prompts[i:i + split_size] for i in range(0, len(prompts), split_size)]

    # Dispatch each sub-batch to a different GPU
    results = []
    for i, sub_batch in enumerate(sub_batches):
        # Move sub_batch to the correct device
        sub_device = f'cuda:{i % num_gpus}'
        print(sub_device)

        inputs = tokenizer(sub_batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        input_ids = inputs["input_ids"].to(sub_device)
        attention_mask = inputs["attention_mask"].to(sub_device)

        # Perform generation using the underlying model on the appropriate device

         # Move model's module to current device
        model.module.to(sub_device)

        generation_config = GenerationConfig(**kwargs)

        # --- intervention code --- #
        def id(head_output, layer_name):
            return head_output

        if interventions == {}:
            intervene = id
            layers_to_intervene = []
        else:
            intervene = partial(intervention_fn, start_edit_location='lt')
            layers_to_intervene = list(interventions.keys())
        
        # --- intervention code --- #
        with torch.no_grad():
            with TraceDict(model.module, layers_to_intervene, edit_output=intervene) as ret:
                generation_output = model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
                output_hidden_states=True,
                output_scores=True,
            )

        # Decode all sequences in the batch
        #outputs = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)

        # Move output back to CPU or main GPU and decode
        generation_output = generation_output.sequences # Assuming you want to gather outputs on cuda:0
        outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        results.extend(outputs)

    return results

def parse_output(output, prompt, tokenizer):

    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    output =output[len(tokenizer.decode(encoded_prompt, skip_special_tokens=False)):]
    #print("Function parse_output: ", output)
    return output

def extract_final_answer(output, cot=True, internal_cot=False):

    if cot: 
        ## Checks if "FINAL ANSWER" is found in the output
        pattern = r"(.*?)(?:final answer)(.*?)(?=\r?\n|$)"
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
    
        #print("_________________")
        #print(out)
        if match:
            matched_text = match.group(2)
            
            #print(matched_text)
            if "yes" in matched_text.lower():
                final_answer = True
                
            elif "no" in matched_text.lower():
                final_answer = False
            
            else:
                #print(matched_text)
                final_answer = "undefined"
                #final_answer = True if "yes" in matched_text.lower() else False ## one line if statement checking if "FINAL ANSWER" is found in the output or not using string comparison
                #results['reasoning'].append({"prompt": prompt_reason, "output": matched_text, "final_answer": final_answer})       

        elif match== None: 
            pattern = r"(.*?)(?:answer)(.*?)(?=\r?\n|$)"
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            #print(out)
            if match:
                matched_text = match.group(2)
                #print(matched_text)
                if "yes" in matched_text.lower():
                    final_answer = True
                    
                elif "no" in matched_text.lower():
                    final_answer = False
                
                else:
                    print(output)
                    #print(matched_text)
                    final_answer = "undefined"

            else:
                #print(output)
                #print(out)
                final_answer = "undefined"
    
    elif internal_cot:
        final_answer = True if "yes" in output.lower() else False

    if final_answer == True: 
        predict = "yes" 

    elif final_answer == "undefined":
        predict = "undefined"   
    else:
        predict = "no"

    return final_answer, predict
    
def load_model(model_name, load_in_8_bit=False): 

    LOAD_8BIT = load_in_8_bit
    BASE_MODEL = model_name
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_4bit=LOAD_8BIT,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if "openchat" in model.config._name_or_path:
        model.generation_config.pad_token_id = 0
        tokenizer.pad_token_id = 0
    elif "Meta-Llama-3-8B-Instruct" in model.config._name_or_path:
        model.generation_config.pad_token_id = 128009#32007
        
        tokenizer.eos_token_id = 128009
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model 

def get_hidden_states(output): 

    #layers = [-1, -8, -14, -16, -18, -24]
    layers = [-1, -8, -16, -24]

    output_hidden_states = {}
    for layer in layers:
        
        ## hidden states[token_generated][layer] --> [batch_size,length_processed_tokens, dimension ]

        ## get hidden state of last generated token [-1] in specific layer [layer] getting the last hidden_states([-1])
        ## Move to CPU and convert to float32 or float64
        last_tensor = output.hidden_states[-1][layer][0][-1].to(device='cpu', dtype=torch.float32)
        # Convert to a list
        tensor_list = last_tensor.tolist()
        # Serialize to JSON
        #json_output = json.dumps(tensor_list)
        output_hidden_states[str(layer)+"_last_token"] = tensor_list

        # ## get hidden state of first generated token ("<s>") [0] in specific layer [layer] getting the last hidden_states([-1:]) 
        # #first_tensor = output.hidden_states[0][layer][0][-1][-1]
        # first_tensor = output.hidden_states[0][layer][0][-1]
        # # Move to CPU and convert to float32 or float64
        # first_tensor =  first_tensor.to(device='cpu', dtype=torch.float32)
        # # Convert to a list
        # tensor_list = first_tensor.tolist()
        # output_hidden_states[str(layer)+"_first_token"] = tensor_list

        # ## get hidden state of first generated token after prefix "GPT-4 assistant:" [8] in specific layer [layer] getting the last hidden_states([-1:]) 
        # first_tensor = output.hidden_states[8][layer][0][-1]
        # # Move to CPU and convert to float32 or float64
        # first_tensor =  first_tensor.to(device='cpu', dtype=torch.float32)
        # # Convert to a list
        # tensor_list = first_tensor.tolist()
        # output_hidden_states[str(layer)+"_after_prefix"] = tensor_list

    return output_hidden_states


def get_attention_scores(output):
    # Specify the layers and heads you are interested in
    layers = [-1, -8, -16, -24]
    #heads = [0, 1, 2, 3, 4, ,5] ### create list for 32 heads
    heads = [num for num in range(32)]
    #heads =  # Example heads, modify as needed

    output_attention_scores = {}
    for layer in layers:
        for head in heads:
            # Get attention score of last generated token [-1] in specific layer [layer] sequence[0] and head [head]
            # AND Move to CPU and convert to float32 or float64
            
            attention_tensor = output.attentions[-1][layer][0][head][-1].to(device='cpu', dtype=torch.float32)

            # Convert to a list
            tensor_list = attention_tensor.tolist()

            # Store in the dictionary
            output_attention_scores[f"layer_{layer}_head_{head}_last_token"] = tensor_list

    return output_attention_scores


def calculate_sequence_uncertainty(output):

    scores= []

    ## calculate the geometric mean of the sequence (becoming an arithmetic mean log-probability) of the most likely token, ignoring that on generation another token can be chosen than the most likeliest. 
    for i in range(len(output.scores)):
        
        ## https://huggingface.co/docs/transformers/v4.36.1/en/internal/generation_utils#transformers.generation.SampleEncoderDecoderOutput
        ## gets probabilities from raw prediction scores of the lm head for each token
        score = torch.softmax(output.scores[i],1).to(device='cpu', dtype=torch.float32)
        ## gets the most likely token score
        score = torch.max(score).item()

        scores.append(score)

    ## https://huggingface.co/docs/transformers/perplexity
    log_scores = np.log(np.array(scores))#.shape
    mean = -np.sum(log_scores)/len(scores)
    return round(mean,2)

def generate_gpu(inputs, tokenizer, model, script_dir, generation_args, stopping_criteria= None, intervention= None):

    ## one line check if cuda is available: 
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    if "openchat" in model.config._name_or_path:
        system_prompt = "GPT4 Correct Assistant:"
        user_prompt = "GPT4 Correct User:"

    elif "Meta-Llama-3-8B-Instruct" in model.config._name_or_path:
        system_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        user_prompt = "<|start_header_id|>user<|end_header_id|>"

    else: 
        system_prompt = "ASSISTANT:"
        user_prompt = "HUMAN:"

    prompt_path = os.path.join(script_dir, inputs["prompt_path"])
    with open(prompt_path, 'r') as f:
        prompt_load = f.read()
  
    tot_results = []

    req = inputs['req']
    counter = 0
    while counter < inputs['consistency']:
        
        ## set up results dict 
        results = {"requirement":req ,'reasoning':[], "split_flag": ""}
        definition = inputs['definition']

        prompt_print = create_prompt_string(prompt_reason=prompt_load, user_prompt=user_prompt, definition = definition, req= req, system_prompt = system_prompt, cot = inputs['cot']) #f"{prompt_reason}\n\n System:{definition} \nRequirement: {req} \nASSISTANT: Let's analyze the given System and requirement step-by-step:" 
        #print(prompt_reason)
        split_size = 1
        subjects = definition.split("\n")[0:50]
        definition = "\n".join(subjects)

        CONTEXT_LENGTH_LLM = 4096
        max_new_length = generation_args['max_new_tokens']
        ## Splits context in case too large to fit in memory
        # while len(tokenizer(prompt_print)['input_ids']) >CONTEXT_LENGTH_LLM-max_new_length: 
        #     split_size += 1
        #     length = len(subjects)//split_size+1
        #     definition_red = "\n".join(subjects[0:length])
        #     #prompt_print = create_prompt_string(prompt_reason=prompt_load, user_prompt=user_prompt, definition = definition, req= req, system_prompt = system_prompt, cot = inputs['cot'])
        #     prompt_print = create_prompt_string(prompt_reason=prompt_print, user_prompt=user_prompt, definition = definition, req= req, system_prompt = system_prompt, cot = inputs['cot'])

        if split_size == 1:

            prompt_reason = create_prompt_string(prompt_reason=prompt_load, user_prompt=user_prompt, definition = definition, req= req, system_prompt = system_prompt, cot = inputs['cot'])
            print(len(tokenizer(prompt_print)['input_ids']))
            #output, complete_output = evaluate(prompt_reason,model = model, tokenizer = tokenizer, device= "cuda", temperature=0.8, top_k = generation_config.top_k, num_beams= 1, top_p = 0.95, do_sample=True, max_new_tokens=MAX_NEW_TOKEN, num_return_sequences=1)
            
            if intervention is not None:
                output, complete_output = eval_intervention()
            output, complete_output = evaluate(prompt_reason, model = model, tokenizer= tokenizer, device = "cuda", stopping_criteria = stopping_criteria, **generation_args)
            
            complete_inputs = output
            #print(output.split("Let's analyze the given System and requirement step-by-step:")[-1])
            #print(output)
            #print(complete_output)
            print(len(complete_output.scores))
            if len(complete_output.scores) == max_new_length:
                results['reasoning'].append({"prompt": prompt_reason, "output": output, "final_answer": False, "predict": "too long", "score": "undefined"})

            scores= torch.softmax(complete_output.scores[-1],1)
            score = round(torch.max(scores).item(),2)#.item()
            print(score)
            sequence_score = calculate_sequence_uncertainty(complete_output)
            hidden_states = get_hidden_states(complete_output)
            # attention_scores = get_attention_scores(complete_output)
            # output = parse_output(output, system_prompt,prompt_reason, internal_cot=inputs["internal_cot"], cot=inputs['cot'])
            
            output = parse_output(output, prompt_reason, tokenizer)
            final_answer, predict = extract_final_answer(output, cot=inputs['cot'], internal_cot=inputs["internal_cot"])

            results['reasoning'].append({"prompt": prompt_reason, "output": output, "final_answer": final_answer, "predict": predict, "score": score, "sequence_uncertainty": sequence_score, "hidden_states": hidden_states, "complete_inputs": complete_inputs})#, "attention_scores": attention_scores})
            
        else: 
            raise Exception("Extracted context is too long.")
    
        tot_results.append(copy.deepcopy(results))
        counter += 1

    return tot_results

#def create_chat_format
def generate_openai(inputs, script_dir): 
    
    counter = 0 

    prompt_path = os.path.join(script_dir, inputs["prompt_path"])
    with open(prompt_path, 'r') as f:
        prompt_load = f.read()

    tot_results = []
    while counter < inputs['consistency']:
        print(f"===== {counter+1} / {inputs['consistency']} =====")
        req = inputs['req']

        definition = inputs['definition']
        prompt_print = create_prompt_string(prompt_reason=prompt_load, user_prompt="", definition = definition, req= req, system_prompt ="", cot = inputs['cot']) #f"{prompt_reason}\n\n System:{definition} \nRequirement: {req} \nASSISTANT: Let's analyze the given System and requirement step-by-step:" 
        
        model =inputs['model']#"gpt-3.5-turbo-1106" #inputs['model']   

        _response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_print}, {"role": "assistant", "content": "Let's think step-by-step:"}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        #stop=["Final Answer: Yes", "Final Answer: No"],#None,
        logprobs=True,
        top_logprobs = 3,
        n=1
        )
        time.sleep(0.5)

        response = _response.model_dump()#dict()

        results = {"requirement":req ,'reasoning':[], "split_flag": ""}
        score = np.exp(response['choices'][0]['logprobs']['content'][-1]['logprob'])
        output = response['choices'][0]['message']['content']
        final_answer, predict = extract_final_answer(output, cot=inputs['cot'], internal_cot=inputs["internal_cot"])

        results['reasoning'].append({"prompt": prompt_print, "output": output, "final_answer": final_answer, "predict": predict, "score": score})
        
        tot_results.append(copy.deepcopy(results))
        counter += 1
        return tot_results

def main(inputs): 

    script_dir = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(script_dir, inputs["input_path"]), "r") as f:
        requirements = json.load(f)

    # Try to load output JSON and process it
    try:    
        output_path = os.path.join(script_dir, inputs["output_path"])
        with open(output_path, "r") as f:
            output_data = json.load(f)

        # Extract req_ids from output_data
        # Assuming each item in the output_data is a dictionary with an 'id' key
        req_ids = set(item['req_id'] for item in output_data if 'req_id' in item)
        print("Skipping out requirements with ids: ", req_ids)
        # Remove these ids from requirements
        # This assumes that requirements is a list of dicts with an 'id' key
        requirements = [req for req in requirements if req.get('req_id') not in req_ids]
        results = output_data
    except FileNotFoundError:
        print(f"Output file {output_path} not found."),
        results = []
    except json.JSONDecodeError:
        print("Error in reading or parsing the JSON file.")


    
    if inputs["method"] == "local":
        #generation_config = GenerationConfig(**extra_args)
        tokenizer, model = load_model(inputs['model'])

        MAX_NEW_TOKENS = 600
        generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
                    "do_sample": True, 
                    "num_beams" : 1, 
                    "num_return_sequences" : 1, 
                    "temperature": 0.3,# 0.8,# 0.8, 
                    "top_p": 0.95,
                #  "min_new_tokens": 256, 
                #"no_repeat_ngram_size": 12, 
                #  "begin_suppress_tokens": [2], 
                    }
        stop_words = ["Yes", "No"]#, "yes", "no"] #"\nYes"

        stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]

        #stop_words_ids = [5592, 5613, 1770]

        #stop_words_ids = [torch.tensor(id) for id in stop_words_ids]
        if inputs["early_stopping"]:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, tokenizer=tokenizer)])

        else: 
            stopping_criteria = None

    for req in tqdm(requirements[:inputs["limit"]]): 
        
        ## define "True" test case 
        inputs['req'] = req['requirement']
        inputs['definition'] = req['definition']

        if inputs["method"] == "local":
            temp = generate_gpu(inputs,tokenizer, model, script_dir, generation_args, stopping_criteria = stopping_criteria)
        elif inputs["method"] == "openai":
            
            def set_environment_variables_from_file(file_path):
                with open(file_path) as file:
                    for line in file:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value

            # Set environment variables from the config file
            set_environment_variables_from_file('config.env')
            OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
            openai.api_key = OPENAI_API_KEY
            
            try: 
                temp = generate_openai(inputs,script_dir)
            
            except Exception as e:
                print(e)
                with open(os.path.join(script_dir, inputs['output_path']), "w") as outfile:
                    json.dump(results, outfile, indent=4)
                raise e

        print(len(temp))
        for i in range(len(temp)):
            #temp[i]["gt"] = "yes"
            temp[i]['req_id'] = req['req_id']
            results.append(temp[i])

    data = results   
    results = []
    for da in data: 

        temp = {}
        temp['req_id'] = da['req_id']
        temp['requirement'] = da['requirement']
        for i in da['reasoning'][0].keys(): 
            temp[i] = da['reasoning'][0][i]

        results.append(temp)
        
        # ## define "Negative" test case 
        # inputs['definition'] = req['red_definition']
        # temp = generate_gpu(inputs,tokenizer, model, script_dir, generation_args, stopping_criteria = stopping_criteria )

        # for i in range(len(temp)):
        #     temp[i]["gt"] = "no"
        #     temp[i]['id'] = req['id']
        #     results.append(temp[i])

    # Assuming script_dir and results are already defined.
    with open(os.path.join(script_dir, inputs['output_path']), "w") as outfile:
        json.dump(results, outfile, indent=4)
        
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process the config file path.')
    parser.add_argument('config_path', help='Path to the config.json file')
    
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config= json.load(file)
    inputs = config 
    main(inputs)

