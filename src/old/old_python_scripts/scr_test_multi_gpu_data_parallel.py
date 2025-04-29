import torch
from torch import nn
from transformers import GPT2Model, GPT2Config

# Define a custom wrapper for the GPT2Model to add logging
class GPT2ModelWithLogging(GPT2Model):
    def forward(self, input_ids, **kwargs):
        # Log the device and input size
        print(f"Running on device: {input_ids.device} with input size: {input_ids.size()}")
        return super().forward(input_ids, **kwargs)

# Configuration for the model
config = GPT2Config()

# Creating a GPT2 model with logging
model = GPT2ModelWithLogging(config)

# Checking if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Moving the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example inputs (batch of dummy data)
batch_size = 8  # Adjust based on your needs and GPU memory
input_ids = torch.randint(50257, (batch_size, 1024)).to(device)  # Random input IDs

# Forward pass
outputs = model(input_ids)

#print(outputs)
