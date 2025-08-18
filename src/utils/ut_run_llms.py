
import torch

from utils.ut_processing_utils import prepare_prompt, parse_output

from transformers import GenerationConfig

from tqdm import tqdm

from baukit.baukit import Trace, TraceDict

from utils.ut_processing_utils import layer_head_to_flattened_idx, flattened_idx_to_layer_head

from einops import rearrange

from functools import partial

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

# def run_llama(args, tokenizer, model, dataset):

#     MAX_NEW_TOKENS = 600
#     generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
#                 "do_sample": True, 
#                 "num_beams" : 1, 
#                 "num_return_sequences" : 1, 
#                 "temperature": args.temperature, #0.8,# 0.8, 
#                 "top_p": 0.95,
#                 }
#                 #"min_new_tokens": 32, 
#                 #"begin_suppress_tokens": [tokenizer.eos_token_id], 
#                 #"no_repeat_ngram_size": 12, 
            
#     results = []
#     counter = 0 

#     for row in tqdm(dataset.iterrows()):
#     #for row in tqdm(df.iterrows()):
        
#         prompt = prepare_prompt(row[1].prompt, tokenizer, args.dataset_name, args.prompt_type)
        
#         #prompt = row[1].prompt
#         #
#         if args.dataset_name != "requirements_data" and args.prompt_type == "ab":
#             prompt = prompt+ " ("

#         #if counter == 30: 
#         #    break

#         #print("Manual evaluation")
#         output = evaluate(
#         prompt,
#         model = model,
#         tokenizer = tokenizer,
#         stopping_criteria = None,
#         device = 'cuda',
#         **generation_args, 
#         )

#         #scores= torch.softmax(output[1].scores[-1],1)
#         #score = round(torch.max(scores).item(),2)#.item()
#         output = parse_output(output[0], prompt, tokenizer)

#         #### --> 
#         id_column = "req_id" if args.dataset_name == "requirements_data" else "data_id"
#         final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
#         if args.dataset_name != "requirements_data" and args.prompt_type != "open_ended":
#             final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
#             gt = row[1]['gt']
#             gt = gt.strip()
#             correct = gt == predict
#             results.append({id_column: row[1][id_column], "prompt": prompt, "output": output, "final_answer": final_answer,"gt": gt, "predict": correct})#, "score": score, "entropy":entropy_norm }),
        
#         elif args.dataset_name == "requirements_data":
#             final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
#             results.append({id_column: row[1][id_column], "prompt": prompt, "output": output, "final_answer": final_answer, "predict": predict})#, "score": score, "entropy":entropy_norm  }),
        
#         else: 
#             results.append({id_column: row[1][id_column], "question": row[1]['question'], "prompt": prompt, "output": output, "answer": output})#, "score": score, "entropy":entropy_norm  }),
        
#         counter += 1
#     return results 

# def run_llama_intervention(args, tokenizer, model, interventions, dataset):
#     print("Running LLM with interventions")
#     num_heads = 32

#     def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):#, add_proj_val_std = args.add_proj_val_std ): 

#             head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
#             for head, direction, proj_val_std in interventions[layer_name]:
#                 #print(head)
#                 #print(direction)
#                 direction_to_add = torch.tensor(direction).to(head_output.device.index)
                
#                 if start_edit_location == 'lt': 
#                     head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add  
                    
#                 else: 
#                     head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
            
#             head_output = rearrange(head_output, 'b s h d -> b s (h d)')
#             return head_output
    
#     def lt_modulated_vector_no_alpha(head_output, layer_name, start_edit_location='lt'): 
            
#         head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
#         for head, direction, proj_val_std in interventions[layer_name]:
#             #print(head)
#             #print(direction)
#             direction_to_add = torch.tensor(direction).to(head_output.device.index)
#             #print(direction_to_add)
#             if start_edit_location == 'lt': 
        
#                 head_output[:, -1, head, :] += proj_val_std * direction_to_add
                
#             else: 

#                 head_output[:, start_edit_location:, head, :] +=  proj_val_std * direction_to_add
        
#         head_output = rearrange(head_output, 'b s h d -> b s (h d)')
#         return head_output
    
#     MAX_NEW_TOKENS = 600
#     generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
#                 "do_sample": True, 
#                 "num_beams" : 1, 
#                 "num_return_sequences" : 1, 
#                 "temperature": args.temperature, #0.8,# 0.8, 
#                 "top_p": 0.95,
#                 #"min_new_tokens": 32, 
#                 #"begin_suppress_tokens": [tokenizer.eos_token_id], 
#                 }
#             #"no_repeat_ngram_size": 12, 
            
    
#     results = []
#     counter = 0 

#     for row in tqdm(dataset.iterrows()):
#     #for row in tqdm(df.iterrows()):
        
#         #print(row[1].prompt)
#         if "<|eot_id|>" not in row[1].prompt:
            
#             #print("prepare_prompt")
#             prompt = prepare_prompt(row[1].prompt, tokenizer, args.dataset_name, args.prompt_type)
#         else:    
#             prompt = row[1].prompt
        
#         #
#         if args.dataset_name != "requirements_data" and args.prompt_type == "ab":
#             prompt = prompt+ " ("
        
#         output = eval_intervention(
#             prompt,
#             model=model,
#             tokenizer=tokenizer,
#             stopping_criteria=None,
#             device='cuda',
#             interventions=interventions,
#             intervention_fn= lt_modulated_vector_add if args.add_or_subtract else lt_modulated_vector_no_alpha,
#             # intervention_fn=lt_modulated_vector_add,  # or lt_modulated_vector_subtract
#             **generation_args,
#         )
            
#         scores= torch.softmax(output[1].scores[-1],1)
#         score = round(torch.max(scores).item(),2)#.item()
#         output = parse_output(output[0], prompt, tokenizer)

#         id_column = "req_id" if args.dataset_name == "requirements_data" else "data_id"
#         final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
#         if args.dataset_name != "requirements_data" and args.prompt_type != "open_ended":
#             final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
#             gt = row[1]['gt']
#             gt = gt.strip()
#             correct = gt == predict
#             results.append({id_column: row[1][id_column], "prompt": prompt, "output": output, "final_answer": final_answer,"gt": gt, "predict": correct, "score": score, }),
        
#         elif args.dataset_name == "requirements_data":
#             final_answer, predict = extract_final_answer_dataset(output, cot=True, internal_cot=False, dataset=args.dataset_name)
#             results.append({id_column: row[1][id_column], "prompt": prompt, "output": output, "final_answer": final_answer, "predict": predict, "score": score }),
        
#         else: 
#             results.append({id_column: row[1][id_column], "question": row[1]['question'], "prompt": prompt, "output": output, "answer": output, "score": score }),
        

#     return results 

def eval_intervention_batch(
    prompts,
    args,
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
    

    #print("Printing from eval_intervention_batch")
    #print(intervention_fn)
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
        intervene = partial(intervention_fn, interventions= interventions, args= args, start_edit_location='lt')
        layers_to_intervene = list(interventions.keys())
    # --- intervention code --- #

    with torch.no_grad():
        
        with TraceDict(model, layers_to_intervene, edit_input=intervene) as ret:
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

def run_llama_intervention_batch(args, tokenizer, model, interventions, intervention_function, dataset, batch_size=2):

    generation_args = {"max_new_tokens":args.max_new_tokens,
                "do_sample": True, 
                "num_beams" : 1, 
                "num_return_sequences" : 1, 
                "temperature": args.temperature,  
                "top_p": 0.95,
                "min_new_tokens": 32, 
                "begin_suppress_tokens": [tokenizer.eos_token_id], 
                }
    
    print("Printing from run_llama_intervention_batch")
    print("Intervention function: ", intervention_function)

    results = []

    #batch_size = 2
    for start_idx in tqdm(range(0, len(dataset), batch_size)):

        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset.iloc[start_idx:end_idx]
        
        prompts = batch.prompt.values.tolist() #row[1].prompt

        prompts_processed = [
            prepare_prompt(prompt, tokenizer, args.dataset_name, args.prompt_type, args.suffix) 
            for prompt in batch.prompt.values.tolist()
        ]

        output = eval_intervention_batch(
        prompts_processed,
        args,
        model = model,
        tokenizer = tokenizer,
        stopping_criteria = None,
        device = 'cuda',
        interventions=interventions, 
        intervention_fn= intervention_function, #lt_modulated_vector_add,
        #intervention_fn=lt_modulated_vector_subtract,
        **generation_args, 
        )

        print("Output: ", output[0])
        # Extract scores (logits) for each generated token
        #logits = torch.stack(output[1].scores, dim=1)  # (batch_size, seq_length, vocab_size)
        # Apply softmax to logits to get probabilities
        #probs = torch.softmax(logits, dim=-1)
        # Compute entropy for each token
        #token_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # (batch_size, seq_length)
        # Normalize entropy by sequence length
        #seq_lengths = token_entropy.size(1)  # Length of each sequence
        #length_normalized_entropy = token_entropy.sum(dim=1) / seq_lengths  # Normalize (batch_size)
        #print(f"Length normalized entropy: {length_normalized_entropy}")
        # Using list comprehension to process the outputs
        new_outputs = [parse_output(out, prompts_processed[i], tokenizer) for i, out in enumerate(output[0])]
        #print(output)
        id_column = "req_id" if args.dataset_name == "requirements_data" else "data_id"
        
        for i, new_output in enumerate(new_outputs):

            row = batch.iloc[i]
            prompt = prompts[i]

            #print("Output: ", new_output)
            results.append({
                        id_column: row[id_column], 
                        "prompt": prompt, 
                        "processed_prompt": prompts_processed[i], 
                        #"complete_output": output[0][i], 
                        "output": new_output, 
                        "gt": row.get("gt", None)
            })
            # if args.prompt_type != "open_ended" else None 
    
    return results


## Run on Multiple GPUs using PyTorch Distributed
import torch.multiprocessing as mp
import torch.distributed as dist
import os 

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Manual Implementation (Problems Demonstrated)
def manual_data_distribution(inputs, rank, world_size, batch_size):
    # Naive splitting of data
    items_per_gpu = len(inputs) // world_size
    start_idx = rank * items_per_gpu
    end_idx = start_idx + items_per_gpu
    
    local_data = inputs[start_idx:end_idx]
    
    # Manual batching
    batches = []
    for i in range(0, len(local_data), batch_size):
        batch = local_data[i:i + batch_size]
        batches.append(batch)
    
    return batches

def cleanup():
    dist.destroy_process_group()

def reshape_flattened_output(flat_list, num_sequences):
    return [
        flat_list[i:i + num_sequences]
        for i in range(0, len(flat_list), num_sequences)
    ]

def run_llama_intervention_batch_parallel_process(rank, world_size, shared_dict, args, tokenizer, model, interventions, dataset, batch_size=2):

    setup(rank, world_size)

    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):#, add_proj_val_std = args.add_proj_val_std ): 

            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                #print(head)
                #print(direction)
                direction_to_add = torch.tensor(direction).to(head_output.device.index)
                
                if start_edit_location == 'lt': 
                    head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add  
                    
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
            
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output
    
    def lt_modulated_vector_no_alpha(head_output, layer_name, start_edit_location='lt'): 
            
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                #print(head)
                #print(direction)
                direction_to_add = torch.tensor(direction).to(head_output.device.index)
                #print(direction_to_add)
                if start_edit_location == 'lt': 
            
                    head_output[:, -1, head, :] += proj_val_std * direction_to_add
                    
                else: 

                    head_output[:, start_edit_location:, head, :] +=  proj_val_std * direction_to_add
            
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output
    
    model.to(rank)

    dataset = dataset.to_dict('records')

    dataloader = manual_data_distribution(dataset, rank, world_size, batch_size)

    num_heads = 32

    MAX_NEW_TOKENS = 600

    num_sequences = args.num_sequences

    if args.temperature != 0: 
        
        generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
                    "do_sample":True, #True, 
                    "num_beams" : 1, 
                    "num_return_sequences" : num_sequences, 
                    "temperature": args.temperature, #0.8,# 0.8, 
                    "top_p": 0.95,
                    "min_new_tokens": 32, 
                    "begin_suppress_tokens": [tokenizer.eos_token_id], 
                    }

    else: 
        
        generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
                    "do_sample":False, #True, 
                    "num_beams" : 1, 
                    "num_return_sequences" : num_sequences, 
                    #"temperature": args.temperature, #0.8,# 0.8, 
                    #"top_p": 0.95,
                    "min_new_tokens": 32, 
                    "begin_suppress_tokens": [tokenizer.eos_token_id], 
                    }
        
    all_local_inputs = []
    
    results = []

    id_column = "req_id" if args.dataset_name == "requirements_data" else "data_id"

    for batch_idx, batch in tqdm(enumerate(dataloader)):

        all_local_inputs.extend(batch)

        prompts = [b['prompt'] for b in batch] # batch.prompt.values.tolist()#row[1].prompt

        prompts = [
            prepare_prompt(prompt, tokenizer, args.dataset_name, args.prompt_type) 
            if "<|eot_id|>" not in prompt 
            else prompt 
            for prompt in prompts#batch.prompt.values.tolist()
        ]
        #print(prompts[0][0:256])
        local_result = eval_intervention_batch(
        prompts,
        model = model,
        tokenizer = tokenizer,
        stopping_criteria = None,
        device = rank,
        interventions=interventions, 
        intervention_fn= lt_modulated_vector_add if args.add_or_subtract else lt_modulated_vector_no_alpha, #lt_modulated_vector_add,
        **generation_args, 
        )

        # Extract scores (logits) for each generated token
        logits = torch.stack(local_result[1].scores, dim=1)  # (batch_size, seq_length, vocab_size)
        #print(logits.shape)
        # Apply softmax to logits to get probabilities
        probs = torch.softmax(logits, dim=-1)
        #print(probs.shape)
        # Compute entropy for each token
        token_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # (batch_size, seq_length)
        #print(f"Token entropy: {token_entropy.shape}")
        # Normalize entropy by sequence length
        #seq_lengths = token_entropy.size(1)  # Length of each sequence
        # Handle sequence lengths if sequences are padded
        #if hasattr(local_result[1], "attention_mask"):  # Assuming an attention mask is available
        if len(local_result[0])>1: 
            # print("Attention mask available")  # For debugging purposes
            # attention_mask = local_result.attention_mask  # (batch_size, seq_length)
            # seq_lengths = attention_mask.sum(dim=1)  # (batch_size,)
            output_sequences = local_result[1].sequences  # (batch_size, seq_length)
            #print(output_sequences.shape)
            total_seq_length = output_sequences.size(1)  # (batch_size, total_seq_length)
            #print("Total sequence length: ", total_seq_length)
            output_seq_length = token_entropy.size(1)  # Number of decoding steps
            #print("Output sequence length: ", output_seq_length)

            # Infer attention mask for output
            pad_token_id = tokenizer.pad_token_id
            #print(pad_token_id)
            #print(output_sequences[0])
            # Slice to get only the output tokens
            generated_output = output_sequences[:, total_seq_length - output_seq_length:]
            #print(tokenizer.batch_decode(generated_output, skip_special_tokens=False))

            output_attention_mask = (generated_output != pad_token_id).long()  # 1 for valid tokens, 0 for padding
            seq_lengths = output_attention_mask.sum(dim=1)  # (batch_size,)
            #print(seq_lengths)
                        

        else:
            seq_lengths = token_entropy.size(1)  # If no padding, all tokens are valid

        
        length_normalized_entropy = token_entropy.sum(dim=1) / seq_lengths  # Normalize (batch_size)
        #print(f"Length normalized entropy: {length_normalized_entropy}")

        #print(f"Without attention mask: {token_entropy.sum(dim=1)/token_entropy.size(1)}")

        ## get max probabilities for each token
        top_probs = torch.topk(probs, k=1, dim=-1)  # (batch_size, seq_length, 1)

        top_probs, top_indices = torch.topk(probs, k=1, dim=-1)
        top_probs = top_probs.detach().tolist()
        #for probs in top_probs:

        #    print(probs[0:5])
        #all_local_results.extend(local_result)

        #local_result = eval_intervention_batch(model, tokenizer, batch, rank)
        #print(local_result[0])
        #print(len(local_result[0]))
        
        #local_result = [parse_output(out, prompts[i], tokenizer) for i, out in enumerate(local_result[0])]

        # Reshape output
        reshaped_output = [local_result[i:i + num_sequences] for i in range(0, len(local_result[0]), num_sequences)]

        reshaped_output = reshape_flattened_output(local_result[0], num_sequences)

        # print(len(reshaped_output))

        length_normalized_entropy =  reshape_flattened_output(length_normalized_entropy, num_sequences)
        
        top_probs = reshape_flattened_output(top_probs, num_sequences)
        
        outputs = []
        for b in range(0,batch_size):
            #print("Batch", b)
            temp = []
            for s in range(0, num_sequences):
                #print("Sequence", s)
                temp.append(parse_output(reshaped_output[b][s], prompts[b], tokenizer))
                
            outputs.append(temp)

        # print(len(outputs[0]))
        #for i, new_output in enumerate(local_result):
        for i, output in enumerate(outputs):
            #print(i)
            row = batch[i] #.iloc

            prompt = prompts[i]

            for j, out in enumerate(output): 
                #print("Num sequences", j)
                #if args.dataset_name != "requirements_data" and args.prompt_type == "ab":
                #    new_output = "(" + new_output
                
                final_answer, predict = extract_final_answer_dataset(out, cot=True, internal_cot=False, dataset=args.dataset_name)
                
                if args.dataset_name != "requirements_data" and args.prompt_type != "open_ended":
                    #final_answer, predict = extract_final_answer_dataset(new_output, cot=True, internal_cot=False, dataset=args.dataset_name)
                    gt = row['gt'].strip()
                    correct = gt == predict
                    results.append({
                        id_column: row[id_column], 
                        "prompt": prompt, 
                        "output": out, 
                        "final_answer": final_answer, 
                        "gt": gt, 
                        "predict": correct,
                        "length_normalized_entropy": length_normalized_entropy[i,j].item(),
                        
                        #"heads": top_heads, 
                        #"score": score
                    })
                
                elif args.dataset_name == "requirements_data":
                    #final_answer, predict = extract_final_answer_dataset(new_output, cot=True, internal_cot=False, dataset=args.dataset_name)
                    #gt = row['final_answer'] if row['correct'] else not row['final_answer']
                    results.append({
                        id_column: row[id_column], 
                        "prompt": prompt, 
                        "output": out, 
                        "final_answer": final_answer, 
                        "predict": predict, 
                        "length_normalized_entropy": length_normalized_entropy[i][j].item(), 
                        "probs": [p[0] for p in top_probs[i][j]],
                    })
                
                else: 
                    results.append({
                        id_column: row[id_column], 
                        "question": row['question'], 
                        "prompt": prompt, 
                        "output": out, 
                        "answer": out, 
                        "length_normalized_entropy": length_normalized_entropy[i,j].item(),
                    })

    results_dict = {
        "rank": rank,
        "local_inputs": all_local_inputs,
        "local_results": results}
    
    shared_dict[rank] = results_dict

    cleanup()

def run_llama_intervention_batch_parallel(args, tokenizer, model, interventions, dataset, batch_size=1):

    world_size = torch.cuda.device_count()

    manager = mp.Manager()

    shared_dict = manager.dict()  # Shared dictionary

    mp.spawn(
        run_llama_intervention_batch_parallel_process,
        args=(world_size,shared_dict, args, tokenizer, model, interventions, dataset, batch_size),
        nprocs=world_size,
        join=True
    )
    
    # Display results after all processes complete
    results = dict(shared_dict)
    overall_results = []
    for rank, result in results.items():
        
        overall_results.extend(result["local_results"])
    #print(overall_results)
    return overall_results


