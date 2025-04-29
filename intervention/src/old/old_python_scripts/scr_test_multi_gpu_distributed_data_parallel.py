import torch
from torch import nn
from transformers import GPT2Model, GPT2Config
from transformers.modeling_utils import unwrap_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
   
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

def setup(rank, world_size):

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

"""
 def unwrap_model(model: nn.Module) -> nn.Module: 
     
     Recursively unwraps a model from potential containers (as used in distributed training). 
  
     Args: 
         model (`torch.nn.Module`): The model to unwrap. 
     
     # since there could be multiple levels of wrapping, unwrap recursively 
     if hasattr(model, "module"): 
         return unwrap_model(model.module) 
     else: 
         return model 
"""


# Define a custom wrapper for the GPT2LMHeadModel to add logging
class GPT2ModelWithLogging(GPT2LMHeadModel):
    def forward(self, input_ids, **kwargs):
        # Log the device and input size
        print(f"Running on device: {input_ids.device} with input size: {input_ids.size()}")
        return super().forward(input_ids, **kwargs)

#tokenizer = AutoTokenizer.from_pretrained("gpt2")


def main(rank, world_size):
    
    print(rank)
    
    # Setup the process for distributed training
    setup(rank, world_size)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load the custom logging model
    model = GPT2ModelWithLogging.from_pretrained("gpt2")

    #device = torch.device("cuda", local_gpu_rank)


    model.to(rank)
    #model = DDP(model, device_ids=[rank])

    # Checking if multiple GPUs are available
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     #model = nn.DataParallel(model)
    #     model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Moving the model to GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    # Example inputs (batch of dummy data)
    batch_size = 8  # Adjust based on your needs and GPU memory
    input_ids = torch.randint(50257, (batch_size, 64)).to(rank)  # Random input IDs

     # Assuming a simple even split for demonstration
    # Calculate the number of data per GPU
    num_data_per_gpu = input_ids.size(0) // world_size
    start_idx = rank * num_data_per_gpu
    end_idx = start_idx + num_data_per_gpu
    input_ids = input_ids[start_idx:end_idx]

    # Forward pass
    outputs = unwrap_model(model).generate(input_ids, max_length= 80)

    cleanup()

#print(outputs)
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    #world_size = 1
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)