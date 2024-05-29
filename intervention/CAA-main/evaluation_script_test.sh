# Declare an array of strings, each string is a JSON array
K_pairs=(
    '["../intervention_results/ai_coordination/open_ended_example_294_304_307_train_extract_multi_heads/results_intervention_35_12_11_13_18_14_27_15_2_.json", "../intervention_results/ai_coordination/open_ended_scoring"]'
    '["../intervention_results/ai_coordination/", "2"]'
)

# Iterate over the array elements
for pair in "${K_pairs[@]}"; do
    # Parse each element of the JSON array
    input=$(echo $pair | jq -r '.[0]')  # Parse first element
    output=$( echo $pair | jq -r '.[1]')  # Parse second element

    echo "Input: $input, Output: $output"
done
