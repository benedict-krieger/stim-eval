#!/bin/bash

set -e 

USER_NAME="ben"
EXP_NAME="adbc23_short"
MODELS=("gerpt2" "gerpt2large" "leo13b" "llammlein120m" 'llammlein1b' 'llammlein7b')

echo "==========================================="
echo "STARTING PIPELINE: $EXP_NAME"
echo "USER: $USER_NAME"
echo "==========================================="

for MODEL in "${MODELS[@]}"; do
    echo ">>> Processing Model: $MODEL"
    
    python run_surprisal.py --user "$USER_NAME" --exp "$EXP_NAME" --llm "$MODEL" --plot
    
    echo ">>> Finished $MODEL"
    echo "-------------------------------------------"
done

FIRST_MODEL=${MODELS[0]}
echo ">>> Merging..."

python run_surprisal.py --user "$USER_NAME" --exp "$EXP_NAME" --llm "$FIRST_MODEL" --merge

echo "==========================================="
echo "PIPELINE COMPLETE"
echo "==========================================="
