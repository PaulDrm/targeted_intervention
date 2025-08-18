
# Check if a configuration file is provided
if [ -n "$1" ]; then
  CONFIG_FILE="$1"
else
  echo "Usage: $0 <config_file>"
  echo "Using default configuration file: $CONFIG_FILE"
fi

# Read and parse the configuration file
CONFIG=$(jq '.' "$CONFIG_FILE")

input=$(echo $CONFIG | jq -r '.input')
output=$(echo $CONFIG| jq -r '.output')   # Parse second element
behaviors=$(echo $CONFIG | jq -r '.behavior')

echo "Input: $input Output: $output Behaviors: $behaviors"

# Define relative path to Python script
PYTHON_SCRIPT="../src/evaluation/scoring_single_score.py"


python "$PYTHON_SCRIPT" --input_path $input --output_path $output --behaviors $behaviors
