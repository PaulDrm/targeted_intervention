import numpy as np
import os 
import json
from tqdm.notebook import tqdm
import re

import pandas as pd

import os

# Change to the desired directory
os.chdir("../")

# Print the current working directory to confirm the change
print("Current Working Directory:", os.getcwd())


from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)

peft_model_path = "./sft_results/requirements_data/llama3_false_positives_1609_KTO_optimised_model/checkpoint-25"  # This should be the path to your PEFT model
config = PeftConfig.from_pretrained(peft_model_path)

peft_model = PeftModel.from_pretrained(base_model, peft_model_path)


import pandas as pd 

folder = "requirements_data"
experiment = "llama3_false_positives_1609_KTO_optimised_model"
#file = "results_epoch_0.96_fold_0_lr_0.0001_seed_42_weight_2.0.json "
file = "results_epoch_2.0_fold_0_lr_0.0001_seed_42_weight_2.0.json "# "test.json"#"tokenizer.json"#"results_epoch_2.0_fold_0_lr_0.0001_seed_42_weight_2.0.json"
df = pd.read_json(f"./sft_results/requirements_data/{experiment}/{file}")

#df
