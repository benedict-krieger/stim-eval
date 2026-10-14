#!/usr/bin/env bash
# Usage (from run.sub):  run.sh <llm> <user> <exp1> [<exp2> ...]
# e.g.                   run.sh leo70b beneschn A21 D19 D21

# -e: stop at the first failing command, -o pipefail: also if a command inside a pipe fails.
set -eo pipefail

LLM="$1"
USERNAME="$2"
shift 2
EXPS=("$@")

PROJECT_DIR="/scratch/beneschn/projects/stim-eval"      
CONDA_SH="/scratch/beneschn/miniconda/etc/profile.d/conda.sh"
SHARED_MODELS="/scratch/common_models"                  # network copy (slow-ish)
LOCAL_MODELS="/local/common_models"                     # node-local copy (fast)

cd "$PROJECT_DIR"
echo "Host: $(hostname)   Start: $(date)"

# ---- Python environment --------------------------------------------------
source "$CONDA_SH"
conda activate ml

# ---- Hugging Face settings -----------------------------------------------
export HF_HOME=/scratch/beneschn/hf_cache

# ---- GPUs ----------------------------------------------------------------
echo "HTCondor gave us: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
source "$PROJECT_DIR/scripts/rename_gpus.sh"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ---- Node-local model copy -----------------------------------------------
# Read the local folder name for this model from llms.yaml (empty if none is set).
MODEL_NAME=$(python -c "import yaml; print(yaml.safe_load(open('config/llms.yaml'))['$LLM'].get('local_dir') or '')")
MODEL_DIR_ARG=()

if [[ -n "$MODEL_NAME" ]]; then
    SRC="$SHARED_MODELS/$MODEL_NAME"
    DST="$LOCAL_MODELS/$MODEL_NAME"
    MARKER="$DST/.copy_complete"     # written only after a copy fully finished
    LOCK="$LOCAL_MODELS/.$MODEL_NAME.lock"

    mkdir -p "$LOCAL_MODELS"

    # flock: if two jobs land on the same node, only one copies, the other waits
    (
        flock -x 200
        if [[ -f "$MARKER" ]]; then
            echo "Local copy found: $DST"
        else
            needed=$(du -sb "$SRC" | cut -f1)
            avail=$(df --output=avail -B1 "$LOCAL_MODELS" | tail -1)
            echo "Model size: $((needed / 1024**3)) GiB, free on /local: $((avail / 1024**3)) GiB"
            if (( avail < needed + 10 * 1024**3 )); then
                echo "Not enough space on /local -> will load directly from $SRC"
                exit 3
            fi
            echo "Copying $SRC -> $DST  ($(date))"
            TMP="$DST.partial.$$"
            rsync -a "$SRC/" "$TMP/"
            rm -rf "${DST:?}"          # remove leftovers of an earlier interrupted copy
            mv "$TMP" "$DST"           # appears under its final name only when complete
            touch "$MARKER"
            echo "Copy done ($(date))"
        fi
    ) 200>"$LOCK" && status=0 || status=$?

    if [[ $status -eq 0 ]]; then
        MODEL_DIR_ARG=(--model_dir "$LOCAL_MODELS")
    elif [[ $status -eq 3 ]]; then
        MODEL_DIR_ARG=(--model_dir "$SHARED_MODELS")
    else
        echo "Copy failed with status $status"; exit $status
    fi
fi

# ---- Run -------------------------------------------------------------------
# -u = unbuffered, so print() output shows up in the .out log while the job runs.
python -u src/run_surprisal.py \
    --llm "$LLM" \
    --user "$USERNAME" \
    --exp "${EXPS[@]}" \
    "${MODEL_DIR_ARG[@]}" \
    --plot

echo "End: $(date)"
