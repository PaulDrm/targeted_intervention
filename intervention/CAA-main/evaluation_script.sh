#output="../intervention_results/ai_coordination/open_ended_scoring" 
#'["../intervention_results/ai_coordination/open_ended_example_294_304_307_train_extract_multi_heads/results_intervention_35_12_11_13_18_14_27_15_2_.json", "../intervention_results/ai_coordination/open_ended_scoring"]',

#     '["../intervention_results/ai_coordination/open_ended_example_294_304_307_test_multi_heads/results_intervention_35_number_heads_4.json", "../intervention_results/ai_coordination/open_ended_scoring"]',
#    '["./ab_cot/results/open_ended/coordinate-other-ais/results_layer=12_multiplier=1.0_behavior=coordinate-other-ais_type=open_ended_use_base_model=False_model_size=7b.json", "../intervention_results/ai_coordination/open_ended_scoring"]'
#    '["./ab_cot/results/open_ended/coordinate-other-ais/results_layer=13_multiplier=1.0_behavior=coordinate-other-ais_type=open_ended_use_base_model=False_model_size=7b.json", "../intervention_results/ai_coordination/open_ended_scoring"]'
#'["../results/test_set_open_ended/iti/results_intervention_35_number_heads_16.json", "../results/test_set_open_ended/iti/"]'
#'["../results/test_set_open_ended/iti_ab/results_intervention_15_number_heads_4.json", "../results/test_set_open_ended/iti_ab/"]'
#'["../results/test_set_open_ended/baseline/results_intervention_0_number_heads_1.json", "../results/test_set_open_ended/baseline/"]'
 # "../results/test_set_open_ended/baseline/"
 # "../results/test_scoring"
 #'["../results/test_set_open_ended/baseline/overall_results.json","../results/test_set_open_ended/baseline/"]',
#'["../results/test_set_open_ended/caa/overall_results.json","../results/test_set_open_ended/caa/"]'
#'["../results/test_set_open_ended/hsi/overall_results.json","../results/test_set_open_ended/hsi/"]'
#'["../results/test_set_open_ended/iti/overall_results.json","../results/test_set_open_ended/iti/"]'
#K_pairs=(
#'["../results/test_set_open_ended/iti_ab/overall_results.json","../results/test_set_open_ended/iti_ab/"]'
#)
K_pairs=('["../intervention_results/sycophancy/test_set_open_ended_baseline/results_intervention_0_number_heads_1.json", "../intervention_results/sycophancy/test_set_open_ended_baseline/"]',
        '["../intervention_results/sycophancy/test_set_open_ended_hsi/results_intervention_125_number_heads_2.json", "../intervention_results/sycophancy/test_set_open_ended_hsi/"]')
behaviors="sycophancy" #"coordinate-other-ais"
for pair in "${K_pairs[@]}"; do
   
    input=$(echo $pair | jq -r '.[0]')  # Parse first element
    output=$(echo $pair | jq -r '.[1]')   # Parse second element
    echo "Input: $input Output: $output"

    python scoring_single_score.py --input_path $input --output_path $output --behaviors $behaviors

done