
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('../')
from intervention.reasoning import eval_intervention, parse_output, extract_final_answer, evaluate

import argparse


import llama
def load_model(model_name):

    LOAD_8BIT = False #True
    BASE_MODEL = model_name

    tokenizer = llama.LlamaTokenizer.from_pretrained(BASE_MODEL)
    model = llama.LlamaForCausalLM.from_pretrained(BASE_MODEL, low_cpu_mem_usage=True, load_in_8bit=LOAD_8BIT, torch_dtype=torch.float16, device_map="auto")

    if "openchat" in model.config._name_or_path:

        model.generation_config.pad_token_id = 0

    return tokenizer, model 

def extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset='requirements_data'):

    if dataset== 'requirements_data':
        final_answer, predict = extract_final_answer(output, cot=cot, internal_cot=internal_cot)
    
    else:#if cot == False: 
        matched_text = output
        #print(matched_text)
        if "(a)" in matched_text.lower():
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='input path')
    parser.add_argument('--output_path', type=str, help='output path')
    parser.add_argument("--model_name", type=str, default='llama_7B', help='model name')
    parser.add_argument("--dataset_name", type=str, default='refusal')
    parser.add_argument('--prompt_type', type=str, default='ab')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    
    args = parser.parse_args()
    df = pd.read_json(args.input_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset_name != "requirements_data":
        id_column = "data_id"
    else: 
        id_column = "req_id"
        correct = [0 if value == "yes" else 1 for value in df.predict.values]
        df.correct = correct

    tokenizer, model = load_model(args.model_name)
    results = []
    counter = 0 
    for row in tqdm(df.iterrows()):

        index = row[0]
        if index % 2 == 0:  # Check if the index is even
          
            prompt = row[1].prompt
            if args.dataset_name != "requirements_data" and args.prompt_type == 'ab':
                prompt = prompt+ " ("

            #if counter ==5: 
            #    break
            
            MAX_NEW_TOKENS = 600
            generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
                        "do_sample": True, 
                        "num_beams" : 1, 
                        "num_return_sequences" : 1, 
                        "temperature": 0.8,# 0.8, 
                        "top_p": 0.95,
                    #  "min_new_tokens": 256, 
                    #"no_repeat_ngram_size": 12, 
                    #  "begin_suppress_tokens": [2], 
                        }

            output = evaluate(
            prompt,
            model = model,
            tokenizer = tokenizer,
            stopping_criteria = None,
            device = 'cuda',
            **generation_args, 
            )

            scores= torch.softmax(output[1].scores[-1],1)
            score = round(torch.max(scores).item(),2)#.item()

            output = parse_output(output[0], prompt, tokenizer)
            #print(output)
            if args.dataset_name != "requirements_data" and args.prompt_type == 'ab':
                output = "(" +output
            #### --> 
            #if args.promp_type == 'ab': 
            final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
            
            #elif args.prompt_type == 'ab_cot':
            #    final_answer, predict = extract_final_answer_dataset(output, cot=False, internal_cot=False, dataset=args.dataset_name)
            

            if args.dataset_name != "requirements_data":
                gt = row[1]['gt']
                gt = gt.strip()
                correct = gt == predict
                results.append({id_column: row[1][id_column], "prompt": prompt, "output": output, "final_answer": final_answer,"gt": gt, "predict": correct, "score": score, }),
            
            else:
                results.append({id_column: row[1][id_column], "prompt": prompt, "output": output, "final_answer": final_answer, "predict": predict, "score": score }),
                

            counter += 1
            if counter % 50 == 0:
                pd.DataFrame(results).to_json(args.output_path, orient='records', indent=True)
    
    pd.DataFrame(results).to_json(args.output_path, orient='records', indent=True)

    #curr_fold_results = pd.DataFrame(results)
if __name__ == "__main__":
    main()   
    