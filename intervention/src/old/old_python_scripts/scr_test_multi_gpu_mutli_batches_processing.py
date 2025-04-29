import torch
import torch.distributed as dist

import torch.multiprocessing as mp

from transformers import GPT2LMHeadModel, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from typing import List, Dict
from tqdm import tqdm 

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process_on_gpu(model: DDP, tokenizer: AutoTokenizer, texts: list, rank: int) -> str:
    """Process text on specific GPU"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,  add_special_tokens=False)
    input_ids = inputs['input_ids'].to(rank)
    
    generation_output = model.generate(
        input_ids=input_ids,
        max_length=100,
        num_return_sequences=1,
        return_dict_in_generate=True,
    )
    
    #outputs = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)
    outputs = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)

    return outputs

def gather_from_all_gpus(local_result: str, rank: int, world_size: int) -> List[str]:
    """Explicitly gather results from all GPUs"""
    # Create tensors for gathering
    max_length = 1024
    local_tensor = torch.zeros(max_length, dtype=torch.long, device=rank)
    local_tokens = torch.tensor([ord(c) for c in local_result], dtype=torch.long, device=rank)
    length = min(len(local_tokens), max_length)
    local_tensor[:length] = local_tokens[:length]
    
    # Gather from all GPUs
    all_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors, local_tensor)
    
    # Convert gathered tensors back to strings
    all_results = []
    for tensor in all_tensors:
        chars = [chr(int(i)) for i in tensor.cpu() if i != 0]
        all_results.append(''.join(chars))
    
    return all_results

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

def main(rank: int, world_size: int, shared_dict: Dict):
    setup(rank, world_size)
    
    # Initialize model on this GPU
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(rank)
    #model = DDP(model, device_ids=[rank])
    
    # Different input for each GPU to demonstrate gathering
    dataset = [
    "Hello from GPU 0",
    "2nd Hello from GPU 0",
    "Greetings from GPU 1",
    "Salutation from GPU 1"
    ]
    
    # num_data_per_gpu = len(inputs_per_gpu) // world_size
    # start_idx = rank * num_data_per_gpu
    # end_idx = start_idx + num_data_per_gpu
    # local_input = inputs_per_gpu[start_idx:end_idx]
    

    batch_size = 2 # Batch size for demonstration
    # # Process on current GPU
    # for start_idx in tqdm(range(0, len(dataset), batch_size)):

    #     end_idx = min(start_idx + batch_size, len(dataset))
    #     batch = dataset.iloc[start_idx:end_idx]

    dataloader = manual_data_distribution(dataset, rank, world_size, batch_size)

        # Process batches
    all_local_inputs = []
    all_local_results = []
    for batch_idx, batch in enumerate(dataloader):
        all_local_inputs.extend(batch)
        local_result = process_on_gpu(model, tokenizer, batch, rank)
        all_local_results.extend(local_result)
    # Gather results from all GPUs
    #all_gpu_results = gather_from_all_gpus(local_result, rank, world_size)
  
    results_dict = None

    results_dict = {
        "rank": rank,
        "local_inputs": all_local_inputs,
        "local_results": all_local_results}
    
    shared_dict[rank] = results_dict

    cleanup()
    #return results_dict
    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    manager = mp.Manager()
    shared_dict = manager.dict()  # Shared dictionary
    mp.spawn(
        main,
        args=(world_size,shared_dict),
        nprocs=world_size,
        join=True
    )

    # Display results after all processes complete
    results = dict(shared_dict)

    overall_results = []
    for rank, result in results.items():
        
        overall_results.extend(result["local_results"])

    print("Overall results:", overall_results)