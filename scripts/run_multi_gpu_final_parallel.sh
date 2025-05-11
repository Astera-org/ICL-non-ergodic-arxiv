#!/bin/bash
# Description: Runs the final training plan for multiple random seeds and K values,
# distributing runs across multiple available GPUs using GNU Parallel.
# Assumes the FULL preprocessed dataset exists locally (e.g., in ./preprocessed_arxiv/).
# Ensure GNU Parallel is installed: sudo apt install parallel (or brew install parallel on macOS)

# Source .env file if it exists
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.

# --- Generate timestamp for this experiment run ---
EXPERIMENT_TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
echo "Starting final experiment run with timestamp: $EXPERIMENT_TIMESTAMP (using GNU Parallel)"

# --- Configuration (Identical to run_multi_gpu_final.sh) ---
MODEL_NAME_OR_PATH="EleutherAI/pythia-70m-deduped"
BATCH_SIZE=16
LEARNING_RATE="0.0003"
LR_SCHEDULE_TYPE="constant_with_warmup"
NUM_WARMUP_STEPS=100
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=32
SEQUENCE_LENGTH=256
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPSILON=1e-8
MAX_GRAD_NORM=1.0
PRECISION="fp16"
MAX_EPOCHS_HARD_LIMIT=5000
STEPS_PER_EVAL_EPOCH=100
CHECKPOINT_INTERVAL=2000
MAX_CHECKPOINTS=12
# GEOM_ALPHA=0.95 # Not directly used by train.py in the parallel version for triggering, but train.py might use it internally if passed
GEOM_BETA=0.95 # For train.py's internal loss-based checkpointing logic
MAX_LOSS_CKPTS=0

# Early Stopping and ReduceLROnPlateau
EARLY_STOPPING_PATIENCE=30
EARLY_STOPPING_DELTA=0.0
REDUCE_LR_FACTOR=1.0
REDUCE_LR_PATIENCE=10
MIN_LR=1e-6
NUM_DATALOADER_WORKERS=2

# W&B Configuration
WANDB_PROJECT="icl-non-ergodic-arxiv"
# WANDB_ENTITY="" # Optional

# Local Output Directory Configuration
LOCAL_TRAINING_OUTPUT_DIR_ROOT="/data/users/adam/checkpoints"
LOCAL_OVERALL_RUN_DIR="${LOCAL_TRAINING_OUTPUT_DIR_ROOT}/run_${EXPERIMENT_TIMESTAMP}_parallel" # Added _parallel to distinguish
mkdir -p "$LOCAL_OVERALL_RUN_DIR"
echo "Overall output directory for this script execution: $LOCAL_OVERALL_RUN_DIR"

# Experiment Grid
K_VALUES_TO_ITERATE=(1 3 5 8 11)
SEED_VALUES_TO_ITERATE=(0 1 2 3)
RUN_SUFFIX_BASE="final_parallel" # Changed base suffix

# S3 Configuration
S3_RESULTS_BUCKET="obelisk-simplex"
S3_RESULTS_PREFIX_ROOT="non-ergodic-arxiv/training_runs"
S3_OVERALL_RUN_PREFIX="${S3_RESULTS_PREFIX_ROOT}/run_${EXPERIMENT_TIMESTAMP}_parallel" # Added _parallel
echo "S3 results root for this script execution: s3://${S3_RESULTS_BUCKET}/${S3_OVERALL_RUN_PREFIX}"

# --- Save Git Repository Metadata for the entire run ---
echo "Gathering git repository metadata for the entire final run..."
METADATA_DIR="${LOCAL_OVERALL_RUN_DIR}/metadata_overall"
mkdir -p "$METADATA_DIR"
METADATA_FILE="${METADATA_DIR}/git_metadata_final_run_${EXPERIMENT_TIMESTAMP}.txt"

echo "# Git Repository Metadata (Final Run - ${EXPERIMENT_TIMESTAMP} - Parallel)" > "$METADATA_FILE"
echo "# Generated at: $(date)" >> "$METADATA_FILE"
echo "# K_VALUES_TO_ITERATE: (${K_VALUES_TO_ITERATE[@]})" >> "$METADATA_FILE"
echo "# SEED_VALUES_TO_ITERATE: (${SEED_VALUES_TO_ITERATE[@]})" >> "$METADATA_FILE"
echo "" >> "$METADATA_FILE"
echo "## Git Information" >> "$METADATA_FILE"
if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "Repository Root: $(pwd)" >> "$METADATA_FILE"
    echo "Remote URL: $(git config --get remote.origin.url)" >> "$METADATA_FILE"
    echo "Current Branch: $(git rev-parse --abbrev-ref HEAD)" >> "$METADATA_FILE"
    echo "Commit Hash: $(git rev-parse HEAD)" >> "$METADATA_FILE"
    echo "Commit Date: $(git log -1 --format=%cd)" >> "$METADATA_FILE"
    echo "Commit Message: $(git log -1 --format=%s)" >> "$METADATA_FILE"
    if git diff --quiet && git diff --staged --quiet; then
        echo "Working Directory: Clean (no uncommitted changes)" >> "$METADATA_FILE"
    else
        echo "Working Directory: Dirty (has uncommitted changes)" >> "$METADATA_FILE"
        echo "" >> "$METADATA_FILE"; echo "## Uncommitted Changes" >> "$METADATA_FILE"
        echo '```diff' >> "$METADATA_FILE"; git diff >> "$METADATA_FILE"; echo '```' >> "$METADATA_FILE"
    fi
else
    echo "Not a git repository or git command not available" >> "$METADATA_FILE"
fi
echo "" >> "$METADATA_FILE"; echo "## W&B Configuration" >> "$METADATA_FILE"
echo "WANDB_PROJECT: $WANDB_PROJECT" >> "$METADATA_FILE"
if [ -n "$WANDB_ENTITY" ]; then echo "WANDB_ENTITY: $WANDB_ENTITY" >> "$METADATA_FILE"; else echo "WANDB_ENTITY: Not specified" >> "$METADATA_FILE"; fi
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_API_KEY: [REDACTED]" >> "$METADATA_FILE"; else echo "WANDB_API_KEY: Not set" >> "$METADATA_FILE"; fi
echo "" >> "$METADATA_FILE"; echo "## Script Hyperparameters (Common)" >> "$METADATA_FILE"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH" >> "$METADATA_FILE"
echo "BATCH_SIZE: $BATCH_SIZE" >> "$METADATA_FILE"
echo "LEARNING_RATE: $LEARNING_RATE" >> "$METADATA_FILE"
echo "NUM_WARMUP_STEPS: $NUM_WARMUP_STEPS" >> "$METADATA_FILE"
echo "MAX_EPOCHS_HARD_LIMIT: $MAX_EPOCHS_HARD_LIMIT" >> "$METADATA_FILE"
echo "STEPS_PER_EVAL_EPOCH: $STEPS_PER_EVAL_EPOCH" >> "$METADATA_FILE"
echo "REDUCE_LR_FACTOR: $REDUCE_LR_FACTOR (1.0 means plateau LR reduction is off)" >> "$METADATA_FILE"
echo "S3_RESULTS_BUCKET: $S3_RESULTS_BUCKET" >> "$METADATA_FILE"
echo "S3_RESULTS_PREFIX_ROOT: $S3_RESULTS_PREFIX_ROOT" >> "$METADATA_FILE"
echo "Overall metadata saved to $METADATA_FILE"

# --- GPU Detection ---
echo "Detecting GPUs..."
set +e
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n 1)
set -e

if ! [[ "$NUM_GPUS_DETECTED" =~ ^[0-9]+$ ]]; then
    echo "Warning: nvidia-smi failed. Assuming 1 GPU."
    NUM_GPUS=1
elif [[ "$NUM_GPUS_DETECTED" -eq 0 ]]; then
    echo "Warning: nvidia-smi reported 0 GPUs. Assuming 1 GPU."
    NUM_GPUS=1
else
    NUM_GPUS=$NUM_GPUS_DETECTED
fi
echo "Will use $NUM_GPUS GPU slot(s) for parallel execution via GNU Parallel."

# --- Directory for individual job logs from GNU Parallel ---
PARALLEL_JOB_LOG_DIR="${LOCAL_OVERALL_RUN_DIR}/parallel_job_script_logs"
mkdir -p "$PARALLEL_JOB_LOG_DIR"
echo "Individual job script logs (managed by parallel) will be in $PARALLEL_JOB_LOG_DIR"

# --- File to store commands for GNU Parallel ---
COMMANDS_FILE="${LOCAL_OVERALL_RUN_DIR}/commands_to_run_parallel.txt"
rm -f "$COMMANDS_FILE" # Clear if exists from a previous run

# --- Build the list of train.py commands ---
echo "Generating commands for train.py..."
for K_VAL in "${K_VALUES_TO_ITERATE[@]}"; do
  for SEED_VAL in "${SEED_VALUES_TO_ITERATE[@]}"; do
    FORMATTED_LR=$(printf "%.0e" "$LEARNING_RATE" | sed 's/e+*0*/e/' | sed 's/e-0*/e-/')
    # This JOB_SPECIFIC_RUN_SUFFIX matches the one train.py will use to create its subdirectory
    JOB_SPECIFIC_RUN_SUFFIX="k${K_VAL}_s${SEED_VAL}_lr${FORMATTED_LR}_bs${BATCH_SIZE}_${RUN_SUFFIX_BASE}_${EXPERIMENT_TIMESTAMP}"

    # Construct the full command for train.py
    # Note: Quoting arguments carefully.
    # The S3_PREFIX for the specific job needs to include the JOB_SPECIFIC_RUN_SUFFIX
    SPECIFIC_S3_PREFIX="${S3_OVERALL_RUN_PREFIX}/${JOB_SPECIFIC_RUN_SUFFIX}"
    
    # The actual python train.py call. Arguments are separated by newlines for readability here,
    # but will be a single line in the command file.
    # Ensure WANDB_ENTITY is passed if set using --wandb_entity
    WANDB_ENTITY_ARG=""
    if [ -n "$WANDB_ENTITY" ]; then
        WANDB_ENTITY_ARG="--wandb_entity \"$WANDB_ENTITY\""
    fi

    COMMAND_LINE="python train.py \\
        --model_name_or_path \"$MODEL_NAME_OR_PATH\" \\
        --k \"$K_VAL\" \\
        --seed \"$SEED_VAL\" \\
        --batch_size \"$BATCH_SIZE\" \\
        --sequence_length \"$SEQUENCE_LENGTH\" \\
        --learning_rate \"$LEARNING_RATE\" \\
        --lr_scheduler_type \"$LR_SCHEDULE_TYPE\" \\
        --num_warmup_steps \"$NUM_WARMUP_STEPS\" \\
        --weight_decay \"$WEIGHT_DECAY\" \\
        --gradient_accumulation_steps \"$GRADIENT_ACCUMULATION_STEPS\" \\
        --adam_beta1 \"$ADAM_BETA1\" \\
        --adam_beta2 \"$ADAM_BETA2\" \\
        --adam_epsilon \"$ADAM_EPSILON\" \\
        --max_grad_norm \"$MAX_GRAD_NORM\" \\
        --precision \"$PRECISION\" \\
        --epochs \"$MAX_EPOCHS_HARD_LIMIT\" \\
        --steps_per_eval_epoch \"$STEPS_PER_EVAL_EPOCH\" \\
        --checkpoint_interval_steps \"$CHECKPOINT_INTERVAL\" \\
        --max_step_checkpoints \"$MAX_CHECKPOINTS\" \\
        --max_loss_ckpts \"$MAX_LOSS_CKPTS\" \\
        --geom_beta \"$GEOM_BETA\" \\
        --early_stopping_patience \"$EARLY_STOPPING_PATIENCE\" \\
        --early_stopping_delta \"$EARLY_STOPPING_DELTA\" \\
        --reduce_lr_factor \"$REDUCE_LR_FACTOR\" \\
        --reduce_lr_patience \"$REDUCE_LR_PATIENCE\" \\
        --min_lr \"$MIN_LR\" \\
        --num_workers \"$NUM_DATALOADER_WORKERS\" \\
        --output_dir \"$LOCAL_OVERALL_RUN_DIR\" \\
        --run_suffix \"$JOB_SPECIFIC_RUN_SUFFIX\" \\
        --wandb_project \"$WANDB_PROJECT\" \\
        $WANDB_ENTITY_ARG \\
        --wandb_run_name \"$JOB_SPECIFIC_RUN_SUFFIX\" \\
        --upload_results_to_s3 \\
        --s3_bucket \"$S3_RESULTS_BUCKET\" \\
        --s3_prefix \"${SPECIFIC_S3_PREFIX}\" \\
        --delete_local_checkpoints_after_s3_upload \\
        --num_best_ema_val_checkpoints 5 \\
        --eval_dataset_multiplier 3"
        # Note: --force_cpu and --disable_wandb are intentionally omitted for default GPU/WandB runs
        
    # Remove newlines and extra spaces for the actual command stored
    CLEANED_COMMAND_LINE=$(echo "$COMMAND_LINE" | tr -s '[:space:]' ' ' | sed 's/ \\ / /g')
    echo "$CLEANED_COMMAND_LINE" >> "$COMMANDS_FILE"
  done
done

TOTAL_JOBS=$(wc -l < "$COMMANDS_FILE")
echo "Generated $TOTAL_JOBS commands for GNU Parallel. Stored in $COMMANDS_FILE"

# --- Run commands using GNU Parallel ---
# Make sure environment variables used by train.py (like AWS credentials, WANDB_API_KEY from .env) are exported.
# The `export $(grep -v '^#' .env | xargs)` at the top should handle .env variables.
# If train.py is not in PATH, use its full path, e.g., "$(pwd)/train.py"

echo "Starting GNU Parallel execution with $NUM_GPUS jobs in parallel..."
echo "Check GNU Parallel's job log at: ${LOCAL_OVERALL_RUN_DIR}/parallel_master_joblog.txt"
echo "Individual train.py outputs will be in subdirectories of: $PARALLEL_JOB_LOG_DIR"

# PARALLEL_JOB_SLOT is 1-indexed
# The subshell ensures that CUDA_VISIBLE_DEVICES is set correctly for each job instance
# and that its output is redirected.
cat "$COMMANDS_FILE" | parallel \
    --jobs "$NUM_GPUS" \
    --joblog "${LOCAL_OVERALL_RUN_DIR}/parallel_master_joblog.txt" \
    --eta \
    --tagstring "[Job {#}/{=टल=}, GPU {=((\${PARALLEL_JOB_SLOT} - 1) % ${NUM_GPUS})=}]" \
    'GPUNUM=$(( (PARALLEL_JOB_SLOT - 1) % NUM_GPUS )); \
     K_SEED_INFO=$(echo {} | grep -o -E "k [0-9]+.*seed [0-9]+" | sed "s/k //g" | sed "s/ seed /_s/g"); \
     LOG_FILENAME="run_${K_SEED_INFO}_gpu${GPUNUM}_job{#}.log"; \
     LOG_FILEPATH="${PARALLEL_JOB_LOG_DIR}/${LOG_FILENAME}"; \
     echo "Starting job {#} on GPU ${GPUNUM}: {}"; \
     echo "Output log: ${LOG_FILEPATH}"; \
     (export CUDA_VISIBLE_DEVICES=${GPUNUM}; {}) > "${LOG_FILEPATH}" 2>&1'

# Explanation of the parallel command:
# --jobs "$NUM_GPUS": Run this many jobs in parallel. Same as -j.
# --joblog: Detailed log from parallel itself about job start/end times, exit codes.
# --eta: Estimate time of completion.
# --tagstring: Custom prefix for each job's output from parallel itself. {#} is job sequence number, {=टल=} is total jobs.
# The command string executed for each line from COMMANDS_FILE (which is represented by {}):
#   GPUNUM=\$(( (PARALLEL_JOB_SLOT - 1) % NUM_GPUS )): Assign GPU ID (0 to NUM_GPUS-1).
#   K_SEED_INFO=...: Extract K and Seed from the command string for a more descriptive log filename.
#   LOG_FILENAME=...: Construct a unique log file name for this specific train.py instance.
#   LOG_FILEPATH=...: Full path to the log file.
#   (export CUDA_VISIBLE_DEVICES=\${GPUNUM}; {}): Sets CUDA_VISIBLE_DEVICES for the command and then executes the command (which is the content of {}).
#                                                The output (stdout & stderr) of this subshell is redirected to LOG_FILEPATH.

echo "--------------------------------------------------"
echo "All $TOTAL_JOBS training jobs have been processed by GNU Parallel."
echo "Check master log: ${LOCAL_OVERALL_RUN_DIR}/parallel_master_joblog.txt"
echo "Check individual job logs in: $PARALLEL_JOB_LOG_DIR"
echo "Check train.py outputs in subdirectories of: $LOCAL_OVERALL_RUN_DIR (e.g., ${LOCAL_OVERALL_RUN_DIR}/k1_s0...)"
echo "Check S3 at s3://${S3_RESULTS_BUCKET}/${S3_OVERALL_RUN_PREFIX}"
echo "--------------------------------------------------"

# Optionally, upload the overall metadata, command file, and parallel logs to S3
OVERALL_ARTIFACTS_S3_PREFIX="${S3_OVERALL_RUN_PREFIX}/_overall_run_artifacts_parallel"
echo "Uploading overall run artifacts (metadata, command file, parallel logs) to s3://${S3_RESULTS_BUCKET}/${OVERALL_ARTIFACTS_S3_PREFIX}"
aws s3 cp --recursive "$METADATA_DIR" "s3://${S3_RESULTS_BUCKET}/${OVERALL_ARTIFACTS_S3_PREFIX}/metadata_overall/"
aws s3 cp "$COMMANDS_FILE" "s3://${S3_RESULTS_BUCKET}/${OVERALL_ARTIFACTS_S3_PREFIX}/commands_to_run_parallel.txt"
aws s3 cp "${LOCAL_OVERALL_RUN_DIR}/parallel_master_joblog.txt" "s3://${S3_RESULTS_BUCKET}/${OVERALL_ARTIFACTS_S3_PREFIX}/parallel_master_joblog.txt"
if [ -d "$PARALLEL_JOB_LOG_DIR" ]; then # Check if the directory was created and has content
    aws s3 cp --recursive "$PARALLEL_JOB_LOG_DIR" "s3://${S3_RESULTS_BUCKET}/${OVERALL_ARTIFACTS_S3_PREFIX}/parallel_job_script_logs/"
fi
echo "Overall artifacts upload complete."

# Note: The GEOM_ALPHA argument was present in the original .sh script but is not an argument to train.py.
# It seems to have been intended for the bash script's internal logic which is now replaced by Parallel.
# train.py uses --geom_beta. If GEOM_ALPHA was meant for train.py, it needs to be added as an argument to train.py.
# Based on train.py's parser, only --geom_beta is used. 