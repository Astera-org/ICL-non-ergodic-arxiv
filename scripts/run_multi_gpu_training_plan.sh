#!/bin/bash
# Description: Runs the training plan for different K values, distributing runs across multiple available GPUs.
# Assumes the FULL preprocessed dataset exists locally (e.g., in ./preprocessed_arxiv/).
# Ensure you have enough compute resources (GPU recommended) and time.

# Source .env file if it exists
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.

# --- Generate timestamp for this experiment run ---
EXPERIMENT_TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
echo "Starting experiment run with timestamp: $EXPERIMENT_TIMESTAMP"

# --- Configuration (adapted from run_full_training_plan.sh) ---
MODEL_NAME_OR_PATH="EleutherAI/pythia-70m-deduped"
BATCH_SIZE=16
LEARNING_RATE="0.0004"   # From user's previous change, now in decimal format for consistent dir naming
LR_SCHEDULE_TYPE="cosine"
NUM_WARMUP_STEPS=200
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=32
SEQUENCE_LENGTH=256
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPSILON=1e-8
MAX_GRAD_NORM=1.0
PRECISION="fp16"         # From user's previous change
TOKEN_BUDGET=2600000000
EPOCHS_FALLBACK=1000     # Fallback epochs if token budget isn't met by train.py
CHECKPOINT_INTERVAL=10000
MAX_CHECKPOINTS=12
GEOM_ALPHA=0.90
GEOM_BETA=0.95
MAX_LOSS_CKPTS=0         # 0 for unlimited, from user's previous change

# W&B Configuration
WANDB_PROJECT="icl-non-ergodic-arxiv"
# WANDB_ENTITY="" # Optional: Your W&B username or team. Uncomment and set if needed.

# Local Output Directory Configuration
# Add the timestamp to local output directory
LOCAL_TRAINING_OUTPUT_DIR="/data/users/adam/checkpoints/run_${EXPERIMENT_TIMESTAMP}" # Define the root for local outputs
mkdir -p "$LOCAL_TRAINING_OUTPUT_DIR"
echo "Local output directory: $LOCAL_TRAINING_OUTPUT_DIR"

# Experiment Grid
K_VALUES=(1 2 3 4 6 8 10 11) # Define K values to run
SEED_VAL=0            # Fixed seed for this plan
RUN_SUFFIX="multi_gpu_plan_seed0" # Suffix for this multi-GPU execution plan

# S3 Configuration
S3_RESULTS_BUCKET="obelisk-simplex" # Replace with your bucket if different
# Add the timestamp to S3 path
S3_RESULTS_PREFIX="non-ergodic-arxiv/training_runs_multi_gpu/run_${EXPERIMENT_TIMESTAMP}" # Differentiated S3 path with timestamp
echo "S3 results will be uploaded to: s3://${S3_RESULTS_BUCKET}/${S3_RESULTS_PREFIX}"

# --- Save Git Repository Metadata ---
echo "Gathering git repository metadata..."

# Create metadata directory
METADATA_DIR="${LOCAL_TRAINING_OUTPUT_DIR}/metadata"
mkdir -p "$METADATA_DIR"

# Generate metadata filename
METADATA_FILE="${METADATA_DIR}/git_metadata_${EXPERIMENT_TIMESTAMP}.txt"

# Initialize the metadata file
echo "# Git Repository Metadata" > "$METADATA_FILE"
echo "# Generated at: $(date)" >> "$METADATA_FILE"
echo "# Run configuration: K_VALUES=(${K_VALUES[@]}), SEED_VAL=$SEED_VAL" >> "$METADATA_FILE"
echo "# Experiment Timestamp: $EXPERIMENT_TIMESTAMP" >> "$METADATA_FILE"
echo "" >> "$METADATA_FILE"

# Capture git information
echo "## Git Information" >> "$METADATA_FILE"
echo "Repository Root: $(pwd)" >> "$METADATA_FILE"

# Check if git is available and the directory is a git repository
if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null; then
    # Get git remote URL
    echo "Remote URL: $(git config --get remote.origin.url)" >> "$METADATA_FILE"
    
    # Get current branch
    echo "Current Branch: $(git rev-parse --abbrev-ref HEAD)" >> "$METADATA_FILE"
    
    # Get current commit hash
    echo "Commit Hash: $(git rev-parse HEAD)" >> "$METADATA_FILE"
    
    # Get commit date
    echo "Commit Date: $(git log -1 --format=%cd)" >> "$METADATA_FILE"
    
    # Get commit message
    echo "Commit Message: $(git log -1 --format=%s)" >> "$METADATA_FILE"
    
    # Check if working directory is clean
    if git diff --quiet && git diff --staged --quiet; then
        echo "Working Directory: Clean (no uncommitted changes)" >> "$METADATA_FILE"
    else
        echo "Working Directory: Dirty (has uncommitted changes)" >> "$METADATA_FILE"
        
        # Optionally capture the diff
        echo "" >> "$METADATA_FILE"
        echo "## Uncommitted Changes" >> "$METADATA_FILE"
        echo '```diff' >> "$METADATA_FILE"
        git diff >> "$METADATA_FILE"
        echo '```' >> "$METADATA_FILE"
    fi
else
    echo "Not a git repository or git command not available" >> "$METADATA_FILE"
fi

# Add W&B information to the metadata
echo "" >> "$METADATA_FILE"
echo "## W&B Configuration" >> "$METADATA_FILE"
echo "WANDB_PROJECT: $WANDB_PROJECT" >> "$METADATA_FILE"
if [ -n "$WANDB_ENTITY" ]; then
    echo "WANDB_ENTITY: $WANDB_ENTITY" >> "$METADATA_FILE"
else
    echo "WANDB_ENTITY: Not specified (using default)" >> "$METADATA_FILE"
fi
if [ -n "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY: [REDACTED - Key is set]" >> "$METADATA_FILE"
else
    echo "WANDB_API_KEY: Not set in environment" >> "$METADATA_FILE"
fi
echo "W&B Run URLs: Will be available in individual run logs" >> "$METADATA_FILE"

echo "" >> "$METADATA_FILE"
echo "## Script Configuration" >> "$METADATA_FILE"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH" >> "$METADATA_FILE"
echo "BATCH_SIZE: $BATCH_SIZE" >> "$METADATA_FILE"
echo "LEARNING_RATE: $LEARNING_RATE" >> "$METADATA_FILE"
echo "LR_SCHEDULE_TYPE: $LR_SCHEDULE_TYPE" >> "$METADATA_FILE"
echo "NUM_WARMUP_STEPS: $NUM_WARMUP_STEPS" >> "$METADATA_FILE"
echo "WEIGHT_DECAY: $WEIGHT_DECAY" >> "$METADATA_FILE"
echo "GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS" >> "$METADATA_FILE"
echo "SEQUENCE_LENGTH: $SEQUENCE_LENGTH" >> "$METADATA_FILE"
echo "PRECISION: $PRECISION" >> "$METADATA_FILE"
echo "TOKEN_BUDGET: $TOKEN_BUDGET" >> "$METADATA_FILE"
echo "S3_RESULTS_BUCKET: $S3_RESULTS_BUCKET" >> "$METADATA_FILE"
echo "S3_RESULTS_PREFIX: $S3_RESULTS_PREFIX" >> "$METADATA_FILE"

echo "Metadata saved to $METADATA_FILE"

# --- Multi-GPU Execution Logic ---
echo "Starting multi-GPU training plan execution (Seed: $SEED_VAL)..."

set +e # Temporarily disable exit on error
# 1. Determine number of GPUs
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n 1)
# NVIDIA_SMI_EXIT_CODE=$? # Capture exit code of the pipeline - Intentionally kept commented or removed if not strictly needed after fix.
set -e # Re-enable exit on error

if ! [[ "$NUM_GPUS_DETECTED" =~ ^[0-9]+$ ]]; then # This line might have been reverted by git pull, ensure it's the robust one
    echo "Warning: nvidia-smi failed or returned non-numeric/empty value ('$NUM_GPUS_DETECTED'). Assuming 1 GPU slot."
    NUM_GPUS=1
elif [[ "$NUM_GPUS_DETECTED" -eq 0 ]]; then
    echo "Warning: nvidia-smi reported 0 GPUs. Assuming 1 GPU slot (may run on CPU if no GPU 0 found by PyTorch)."
    NUM_GPUS=1
else
    NUM_GPUS=$NUM_GPUS_DETECTED
fi
echo "Will attempt to use $NUM_GPUS GPU slot(s) for parallel execution."

# Create output directory for script logs if it doesn't exist
SCRIPT_LOG_DIR="${LOCAL_TRAINING_OUTPUT_DIR}/script_logs"
mkdir -p "$SCRIPT_LOG_DIR"
echo "Main script logs for each K-value run will be stored in $SCRIPT_LOG_DIR"

# --- Job Management ---
K_VALUES_TO_RUN=("${K_VALUES[@]}") # Copy to a modifiable array for queueing

# Associative array to store PIDs of jobs running on each GPU slot. Key: gpu_id, Value: pid
declare -A pids_on_gpu
# Associative array to store K value for the job on a GPU slot. Key: gpu_id, Value: k_val
declare -A k_val_on_gpu

# Initialize all GPU slots as free (pid 0)
for (( i=0; i<$NUM_GPUS; i++ )); do
    pids_on_gpu[$i]=0
    k_val_on_gpu[$i]=""
done

completed_k_count=0
total_k_to_process=${#K_VALUES[@]}

echo "Total K values to process: $total_k_to_process"

while [[ $completed_k_count -lt $total_k_to_process ]]; do
    # Attempt to launch new jobs on free GPUs
    for gpu_id in $(seq 0 $(($NUM_GPUS - 1)) ); do
        current_pid=${pids_on_gpu[$gpu_id]}

        # Check if GPU slot is free (current_pid is 0) or the job on it has finished
        if [[ $current_pid -eq 0 ]] || ! kill -0 "$current_pid" 2>/dev/null; then
            if [[ $current_pid -ne 0 ]]; then # A job just finished
                echo "INFO: Job for K=${k_val_on_gpu[$gpu_id]} (PID $current_pid) on GPU $gpu_id has completed."
                ((completed_k_count++))
                echo "INFO: Completed $completed_k_count out of $total_k_to_process K-value runs."
                pids_on_gpu[$gpu_id]=0 # Mark as free
                k_val_on_gpu[$gpu_id]=""
            fi
            
            # If there are K values left to launch from the queue
            if [[ ${#K_VALUES_TO_RUN[@]} -gt 0 ]]; then
                k_val_to_launch="${K_VALUES_TO_RUN[0]}" # Get first K value from queue
                K_VALUES_TO_RUN=("${K_VALUES_TO_RUN[@]:1}") # Remove it from queue

                echo "--------------------------------------------------"
                echo "LAUNCHING: Training for K=$k_val_to_launch on GPU $gpu_id"
                echo "--------------------------------------------------"
                
                JOB_LOG_FILE="${SCRIPT_LOG_DIR}/run_k${k_val_to_launch}_seed${SEED_VAL}_gpu${gpu_id}.log"
                
                # Construct the exact subdirectory name that train.py will use
                # This includes learning rate, batch size, and the full run_suffix passed to train.py
                TRAIN_PY_RUN_SUFFIX="${RUN_SUFFIX}_${EXPERIMENT_TIMESTAMP}"
                TRAIN_PY_SUBDIR_NAME="k${k_val_to_launch}_seed${SEED_VAL}_lr${LEARNING_RATE}_bs${BATCH_SIZE}_${TRAIN_PY_RUN_SUFFIX}"
                
                # Define the path for the metadata file copy within the train.py output directory
                RUN_METADATA_TARGET_PATH="${LOCAL_TRAINING_OUTPUT_DIR}/${TRAIN_PY_SUBDIR_NAME}/git_metadata.txt"
                
                # Ensure parent directory for metadata link exists (train.py will also create this, but mkdir -p is safe)
                mkdir -p "$(dirname "$RUN_METADATA_TARGET_PATH")"
                cp "$METADATA_FILE" "$RUN_METADATA_TARGET_PATH"
                echo "Metadata copied to: $RUN_METADATA_TARGET_PATH"
                
                CMD=(
                  "python" "train.py"
                  "--model_name_or_path" "$MODEL_NAME_OR_PATH"
                  "--k" "$k_val_to_launch"
                  "--seed" "$SEED_VAL"
                  "--batch_size" "$BATCH_SIZE"
                  "--sequence_length" "$SEQUENCE_LENGTH"
                  "--learning_rate" "$LEARNING_RATE"
                  "--lr_scheduler_type" "$LR_SCHEDULE_TYPE"
                  "--num_warmup_steps" "$NUM_WARMUP_STEPS"
                  "--weight_decay" "$WEIGHT_DECAY"
                  "--adam_beta1" "$ADAM_BETA1"
                  "--adam_beta2" "$ADAM_BETA2"
                  "--adam_epsilon" "$ADAM_EPSILON"
                  "--max_grad_norm" "$MAX_GRAD_NORM"
                  "--gradient_accumulation_steps" "$GRADIENT_ACCUMULATION_STEPS"
                  "--precision" "$PRECISION"
                  "--token_budget" "$TOKEN_BUDGET"
                  "--epochs" "$EPOCHS_FALLBACK"
                  "--checkpoint_interval_steps" "$CHECKPOINT_INTERVAL"
                  "--max_step_checkpoints" "$MAX_CHECKPOINTS"
                  "--geom_alpha" "$GEOM_ALPHA"
                  "--geom_beta" "$GEOM_BETA"
                  "--max_loss_ckpts" "$MAX_LOSS_CKPTS"
                  "--wandb_project" "$WANDB_PROJECT"
                  "--run_suffix" "$TRAIN_PY_RUN_SUFFIX" # Pass the combined suffix
                  "--output_dir" "$LOCAL_TRAINING_OUTPUT_DIR" 
                  "--upload_results_to_s3"
                  "--s3_results_bucket" "$S3_RESULTS_BUCKET"
                  "--s3_results_prefix" "$S3_RESULTS_PREFIX"
                )
                if [ -n "$WANDB_ENTITY" ]; then
                  CMD+=("--wandb_entity" "$WANDB_ENTITY")
                fi

                echo "Executing on GPU $gpu_id: ${CMD[@]}"
                echo "Script log for this job: $JOB_LOG_FILE"
                
                ( CUDA_VISIBLE_DEVICES=$gpu_id "${CMD[@]}" ) > "$JOB_LOG_FILE" 2>&1 &
                
                pids_on_gpu[$gpu_id]=$!
                k_val_on_gpu[$gpu_id]=$k_val_to_launch
                echo "INFO: Launched K=$k_val_to_launch on GPU $gpu_id with PID ${pids_on_gpu[$gpu_id]}."
            fi
        fi
    done

    # If not all K-values have completed, sleep before checking again
    if [[ $completed_k_count -lt $total_k_to_process ]]; then
        sleep 10 # Check status every 10 seconds
    fi
done

echo "--------------------------------------------------"
echo "All $total_k_to_process K-value training runs have completed."
echo "Multi-GPU training plan execution finished."
echo "Script logs for each launch command are in: $SCRIPT_LOG_DIR"
echo "Individual \`train.py\` outputs and logs are in subdirectories under $LOCAL_TRAINING_OUTPUT_DIR"
echo "Git repository metadata saved to: $METADATA_FILE"
echo "S3 path: s3://${S3_RESULTS_BUCKET}/${S3_RESULTS_PREFIX}"
echo "--------------------------------------------------"

exit 0 