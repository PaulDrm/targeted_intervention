# Attention Head Specific Activation Intervention for LLMs

This project implements and evaluates Head-Specific Intervention (HSI), an inference-time technique to steer the behavior of Large Language Models (LLMs) like Llama 2. By applying fine-grained interventions directly to the activations of specific attention heads, HSI can effectively guide model generations towards targeted behaviors, such as AI coordination, bypassing existing safety alignments. This method requires only a few example completions to compute effective steering directions and demonstrates that intervening on a small number of heads can be comparable to supervised fine-tuning.

## Papers

For more details, see our papers:
- [Head-Specific Intervention Can Induce Misaligned AI Coordination in Large Language Models](https://arxiv.org/abs/2502.05945).
- [Inference-Time Intervention in Large Language Models for Reliable Requirement Verification](https://arxiv.org/abs/2503.14130).

## Requirements

- Python 3.11
- Packages listed in `requirements.txt` 

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PaulDrm/targeted_intervention
    cd targeted_intervention
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install the project source code (editable mode):**
    ```bash
    pip install -e .
    ```

## Usage

Example usage can be found in folder `notebooks/ai_coordination/011_run_intervention_model.ipynb`