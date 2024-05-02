#!/bin/bash

# Load the config
CONFIG=$(jq '.' experiments/ai_coordination/config_train_set_tests_intervention_294.json)

# Get the K_pairs as a single JSON array of arrays
K_pairs=$(echo $CONFIG | jq '.K_pairs')

# Pass K_pairs to the Python script
python_script="scr_test_multiple_heads_parsing.py"
python $python_script --list_of_heads "$K_pairs"
