from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

# KTO imports
from trl import (SFTConfig, 
                DPOConfig,
                KTOConfig, 
                ModelConfig, 
                get_peft_config, 
                setup_chat_format
)

@dataclass
class BaseArgs:
    """
    Arguments that are always needed, no matter which trainer is chosen.
    """
    model_name_or_path: str = field(
        default="",
        metadata={"help": "Name of or path to the base model."}
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "Directory where model and results will be saved."}
    )
    trainer_type: str = field(
        default="KTO",
        metadata={
            "help": "Which trainer to use. Possible values: 'KTO', 'SFT', or 'DPO'."
        }
    )

def main():
    # 1. Parse ONLY the base arguments first
    parser = HfArgumentParser(BaseArgs)
    base_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # 2. Pick which config to parse next
    if base_args.trainer_type.upper() == "KTO":
        parser = HfArgumentParser(KTOConfig)
        config_args, = parser.parse_args_into_dataclasses(args=remaining_args)
        
        # Now you have KTOConfig as `config_args`
        # If you import KTOTrainer from trl, for example:
        from trl import KTOTrainer
        
        # This is the trainer you'll build:
        TrainerClass = KTOTrainer
        trainer_config = config_args  # name it something explicit if you like

    elif base_args.trainer_type.upper() == "SFT":
        parser = HfArgumentParser(SFTConfig)
        config_args, = parser.parse_args_into_dataclasses(args=remaining_args)
        
        from trl import SFTTrainer
        TrainerClass = SFTTrainer
        trainer_config = config_args

    elif base_args.trainer_type.upper() == "DPO":
        parser = HfArgumentParser(DPOConfig)
        config_args, = parser.parse_args_into_dataclasses(args=remaining_args)
        
        from trl import DPOTrainer
        TrainerClass = DPOTrainer
        trainer_config = config_args

    else:
        raise ValueError(f"Unknown trainer type: {base_args.trainer_type}")
    
    print(f"Trainer config: {trainer_config}")

if __name__ == "__main__":
    main()
