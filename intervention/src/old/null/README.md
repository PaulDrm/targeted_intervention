---
base_model: meta-llama/Meta-Llama-3-8B-Instruct
library_name: peft
license: llama3
tags:
- trl
- kto
- generated_from_trainer
model-index:
- name: 'null'
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/pauld/huggingface/runs/5ep5fter)
# null

This model is a fine-tuned version of [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6004
- Eval/rewards/chosen: 0.0713
- Eval/logps/chosen: -174.6075
- Eval/rewards/rejected: 0.0986
- Eval/logps/rejected: -217.2799
- Eval/rewards/margins: -0.0273
- Eval/kl: 0.7783

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |        |
|:-------------:|:------:|:----:|:---------------:|:------:|
| 0.5651        | 0.9677 | 15   | 0.6026          | 0.1513 |
| 0.5618        | 2.0    | 31   | 0.5999          | 0.3742 |
| 0.5484        | 2.9677 | 46   | 0.6006          | 0.6711 |
| 0.5466        | 4.0    | 62   | 0.6003          | 0.8158 |
| 0.6017        | 4.8387 | 75   | 0.6004          | 0.7783 |


### Framework versions

- PEFT 0.11.1
- Transformers 4.42.2
- Pytorch 2.2.0
- Datasets 2.20.0
- Tokenizers 0.19.1