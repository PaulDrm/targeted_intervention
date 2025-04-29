import sys
sys.path.append('../')
from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers import EarlyStoppingCallback
from intervention.reasoning import load_model, eval_intervention, eval_intervention_batch, parse_output, extract_final_answer, evaluate 
from trl import KTOConfig, KTOTrainer, ModelConfig, setup_chat_format
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm 
from ut_evaluation_utils import precision_recall_consistency
import os 
import json
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainerCallback

import pandas as pd 

from ut_processing_utils import process_data

def custom_evaluate(model, tokenizer, eval_dataset, epoch, output_dir, temperature, fold, lr, seed, undes_weight):
    model.eval()
    MAX_NEW_TOKENS = 600
    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": temperature,
        "top_p": 0.95,
    }
    
    results = []
    for row in tqdm(eval_dataset.iterrows()):
        prompt = row[1]['prompt']
        output = evaluate(prompt, model, tokenizer, **generation_args)
        output = parse_output(output[0], prompt, tokenizer)
        cot = True
        internal_cot = False
        final_answer, predict = extract_final_answer(output, cot=cot, internal_cot=internal_cot)
        gt = row[1]['final_answer'] if row[1]['correct'] else not row[1]['final_answer']
        results.append({
            'id': row[1][id_column],
            'req_id': row[1][id_column],
            "prompt": prompt,
            "output": output,
            "final_answer": final_answer,
            "predict": predict,
            "gt": gt,
            "epoch": epoch,
        })
    
    df_results = pd.DataFrame(results)
    precision, recall = precision_recall_consistency(df_results)
    metrics = {
        "epoch": epoch,
        "precision": precision,
        "recall": recall,
        "fold": fold,
    }
    print(metrics)
    save_results(df_results, metrics, output_dir, epoch, fold, lr, seed, undes_weight)
    return metrics

def save_results(df_results, metrics, output_dir, epoch, fold, lr, seed, undes_weight):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results DataFrame
    results_file = os.path.join(output_dir, f"results_epoch_{epoch}_fold_{fold}_lr_{lr}_seed_{seed}_weight_{undes_weight}.json ")
    df_results.to_json(results_file, orient='records', indent=4)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    # Also save current epoch metrics separately for easy access
    current_metrics_file = os.path.join(output_dir, f"metrics_epoch_{epoch}_fold_{fold}_lr_{lr}_seed_{seed}_weight_{undes_weight}.json")
    with open(current_metrics_file, 'w') as f:
        json.dump(metrics, f)

@dataclass
class ScriptArguments:
    output_path: str = field(metadata={"help": "The output directory"})
    num_fold: int = field(default=2, metadata={"help": "Number of folds for cross-validation"})
    temperature: float = field(default=0.3, metadata={"help": "Sampling temperature"})
    rmsprop_alpha: float = field(default=0.99, metadata={"help": "RMSprop decay rate"})
    rmsprop_eps: float = field(default=1e-8, metadata={"help": "RMSprop epsilon"})
    #gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    #learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate"})
    #weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})

class CustomKTOTrainer(KTOTrainer):
    def create_optimizer(self):
        """
        Override the default optimizer creation to use SGD
        """
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if n in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if n not in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate
                # momentum=0.0  # You can set momentum if desired
            )
        
        print(f"Setting up optimizer: SGD with learning rate {self.args.learning_rate} and weight decay {self.args.weight_decay}...")
        return self.optimizer

class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, output_dir, fold, eval_steps=1, temperature=0.8, learning_rate=None, seed=42, undes_weight=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.temperature = temperature
        self.fold = fold
        self.learning_rate = learning_rate
        self.seed = seed
        self.undes_weight = undes_weight
    #def on_train_begin(self, args, state, control, **kwargs):
        #self.trainer = kwargs.get('trainer', None)

    def on_epoch_end(self, args, state, control, model, **kwargs):
        #logger.info(f"Epoch {state.epoch} has ended")
        metrics = custom_evaluate(model, self.tokenizer, self.eval_dataset, 
                                   state.epoch, self.output_dir, self.temperature, self.fold, self.learning_rate, self.seed, self.undes_weight)


if __name__ == "__main__":
    
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    
    args, kto_args, model_args = parser.parse_args_into_dataclasses()
    
    print(f"Arguments: {args}")
    print(f"Model arguments: {model_args}")
    print(f"KTO arguments: {kto_args}")
    
    print(f"Model output_dir: {kto_args.output_dir}")
    kto_args.output_dir = args.output_path
    args.input_path = "../datasets/requirements_data/dataframe_llama3b_cot_moon_12072024_attentions_false_positives_corout.json"
    
    df = pd.read_json(args.input_path)
    id_column = "req_id"
    column = "attentions"

    # Set seeds
    torch.manual_seed(kto_args.seed)
    np.random.seed(kto_args.seed)
    torch.cuda.manual_seed_all(kto_args.seed)

    args.consistency_factor = 5

    ## Process the data
    index_dic, separated_activations, separated_labels, reqs_order = process_data(df, id_column)

    ## Set up random number generator
    rng = np.random.default_rng(kto_args.seed)

    # Create indices for all samples
    all_indices = np.arange(len(reqs_order))

    # Shuffle indices
    #rng.shuffle(all_indices)

    # Split indices into folds
    if args.num_fold == 1:
        train_size = int(len(all_indices) * 0.6)
        fold_indices = [
            all_indices[train_size:], # 40% for validation/test
            all_indices[:train_size]  # 60% for training   
        ]

        gts = pd.read_json("../datasets/requirements_data/moonbase_requirements_gt.json")

        gt_labels = gts[gts['req_id'].isin(reqs_order)]['gt'].values

        num_folds = 2 #args.num_fold# Number of folds

        # Organize indices by class
        class_indices = {}
        for idx, label in zip(all_indices, gt_labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Shuffle indices within each class
        for indices in class_indices.values():
            rng.shuffle(indices)

        # Split indices into stratified folds
        stratified_folds = [[] for _ in range(num_folds)]
        for indices in class_indices.values():
            fold_size = len(indices) // num_folds
            remainder = len(indices) % num_folds

            start = 0

            for i in range(num_folds):
                end = start + fold_size #+ (1 if i < remainder else 0)
                stratified_folds[i].extend(indices[start:end])
                start = end

        fold_indices = stratified_folds
        #print(stratified_folds[1])
        print(gts[gts['req_id'].isin([reqs_order[i] for i in stratified_folds[0]])]['gt'].value_counts())
        print(gts[gts['req_id'].isin([reqs_order[i] for i in stratified_folds[1]])]['gt'].value_counts())

    else:
        fold_size = len(all_indices) // args.num_fold
        fold_indices = [all_indices[i:i+fold_size] for i in range(0, len(all_indices), fold_size)]

    # formatted_dataset = dataset.map(format_dataset)
    from datasets import Dataset
    # Assuming you have a pandas DataFrame called 'df'

    # Iterate over all folds
    for fold in range(args.num_fold):

        print(f"\nFold {fold + 1}/{args.num_fold}")
        #if fold == 0:
        #    continue

        # Use current fold as validation set, and the rest as training set
        test_idxs = fold_indices[fold]
        #train_idxs = np.concatenate([fold_indices[i] for i in range(args.num_fold) if i != fold])
        train_idxs = np.concatenate([fold_indices[i] for i in range(len(fold_indices)) if i != fold])
        
        
        val_idxs = train_idxs
        print(f"Train set size: {len(train_idxs)}")
        print(f"Validation set size: {len(val_idxs)}")
        print(f"Test set size: {len(test_idxs)}")
        
        # Create datasets
        train_index_expanded = np.concatenate([list(index_dic.values())[i] for i in train_idxs])
        val_index_expanded = np.concatenate([list(index_dic.values())[i] for i in val_idxs])
        test_index_expanded = np.concatenate([list(index_dic.values())[i] for i in test_idxs])
        train_set_idxs = train_idxs
        val_set_idxs = test_idxs # val_idxs
        print(train_set_idxs)
        print(val_set_idxs)

        train_set = df.loc[train_index_expanded]
        print(train_set.shape)
        val_set = df.loc[val_index_expanded]
        test_set = df.loc[test_index_expanded]
        print(test_set.shape)
        #break

    kto_dataset_dict = {
        'prompt': train_set['prompt'].tolist(),
        'completion': train_set['output'].tolist(),
        'label': train_set['correct'].tolist()}

    custom_device_map = {}

    for i in range(32):  # Assuming your model has 80 layers, adjust as needed
        if i < 14:
            custom_device_map[f"model.layers.{i}"] = 0  # Assign to GPU 0
        else:
            custom_device_map[f"model.layers.{i}"] = 1  # Assign to GPU 1

    # Include any other specific mappings if needed (e.g., for output layers)
    custom_device_map["model.embed_tokens.weight"]=0
    custom_device_map["model.norm.weight"] = 1       # Assign final normalization to GPU 1
    custom_device_map["lm_head"] = 1    # Assign output head to GPU 1

    # Load model for full fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        #device_map="auto",
        device_map=custom_device_map, 
        use_cache=not kto_args.gradient_checkpointing
    )

    # Enable gradient checkpointing for memory efficiency
    if kto_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_dict(kto_dataset_dict)
    #train_dataset = Dataset.from_pandas(train_dataset)

            # Get max prompt length
    max_prompt_length = max(len(tokenizer.encode(prompt)) for prompt in kto_dataset_dict['prompt'])

    # Get max sequence length (prompt + completion combined)
    max_sequence_length = max(
        len(tokenizer.encode(prompt + completion)) 
        for prompt, completion in zip(kto_dataset_dict['prompt'], kto_dataset_dict['completion'])
    )

    print(f"Max prompt length: {max_prompt_length}")
    print(f"Max sequence length: {max_sequence_length}")


    #Define a function to concatenate prompt and output
    def concatenate_prompt_output(example):
        full_prompt = example['complete_inputs']#example['prompt'] + example['output']
        completion = example['output']
        label = example['correct']
        return {'text': full_prompt, 'completion': completion, 'label': label}

    # # Apply the function to each example in the dataset
    # train_dataset = train_dataset.map(concatenate_prompt_output)

    test_set.reset_index(drop=True, inplace=True)
    indexes = [test_set[test_set['req_id'] == req_id].index[0] for req_id in test_set.req_id.unique()]
    
    print(indexes)
    print(test_set.req_id.unique())

    # Repeat the list 10 times
    repeated_indexes = indexes * args.consistency_factor
    print(repeated_indexes)
    #test_set = train_set.loc[train_set.index.repeat(n)]
    test_set = test_set.loc[repeated_indexes]
    print(test_set.shape)

    val_set = test_set#[['prompt', 'output']]

    eval_dataset = Dataset.from_pandas(val_set)
    eval_dataset = eval_dataset.map(concatenate_prompt_output)

    output_dir_results = args.output_path#"../sft_results/requirements_data/llama3_false_positives_0609_stratified_lr_sweep" #args.output_path #
    # If we are aligning a base model, we use ChatML as the default template
    #if tokenizer.chat_template is None:
    #    model, tokenizer = setup_chat_format(model, tokenizer)

    #Load the dataset
    #dataset = load_dataset(args.dataset_name)

    # Apply chat template
    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
        return example

    #formatted_dataset = dataset.map(format_dataset)
    #print(96*"=")
    #print(f"Train set example: {formatted_dataset['train'][0]}")
    #print(96*"=")
    print("Train example....")
    print(train_dataset[0])

    # Update training configuration
    kto_args.gradient_accumulation_steps = 8
    kto_args.per_device_train_batch_size = 2
    kto_args.per_device_eval_batch_size = 2
    kto_args.eval_strategy = "epoch"
    kto_args.save_strategy = "epoch"
    kto_args.max_grad_norm = 1.0
    kto_args.use_liger_kernel=True
    #kto_args.rmsprop_alpha = args.rmsprop_alpha
    #kto_args.rmsprop_eps = args.rmsprop_eps

    kto_args.optim = "sgd"#"rmsprop"

    
    kto_args.max_length = 1536
    kto_args.max_prompt_length = 1024

    print("Set max length to ")
    print(kto_args.max_length)
    print("Set max prompt length to ")
    print(kto_args.max_prompt_length)


    # Initialize trainer with custom callback and RMSprop optimizer
    custom_callback = CustomEvalCallback(
        test_set, 
        tokenizer, 
        args.output_path,
        fold=0,
        temperature=args.temperature,
        learning_rate=kto_args.learning_rate,
        seed=kto_args.seed,
        undes_weight=kto_args.undesirable_weight
    )

    # 
    # CustomKTOTrainer(
    trainer =KTOTrainer(
        model=model,
        args=kto_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[custom_callback],
        )

    # Train and save the model
    trainer.train()
    trainer.save_model(kto_args.output_dir)
    trainer.push_to_hub()
