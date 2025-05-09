#!/bin/bash
# Description: Runs the training plan for different K values, distributing runs across multiple available GPUs.
# Assumes the FULL preprocessed dataset exists locally (e.g., in ./preprocessed_arxiv/).
# Ensure you have enough compute resources (GPU recommended) and time.

# Source .env file if it exists
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.

echo "Debug: Script started, set -e and set -o pipefail are active."

# --- Configuration (adapted from run_full_training_plan.sh) --- 
MODEL_NAME_OR_PATH="EleutherAI/pythia-70m-deduped"
BATCH_SIZE=16
LEARNING_RATE=4e-4       # From user's previous change
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
LOCAL_TRAINING_OUTPUT_DIR="/data/users/adam/checkpoints" # Define the root for local outputs

# Experiment Grid
K_VALUES=(1 2 3 4 6 8 10 11) # Define K values to run
SEED_VAL=0            # Fixed seed for this plan
RUN_SUFFIX="multi_gpu_plan_seed0" # Suffix for this multi-GPU execution plan

# S3 Configuration
S3_RESULTS_BUCKET="obelisk-simplex" # Replace with your bucket if different
S3_RESULTS_PREFIX="non-ergodic-arxiv/training_runs_multi_gpu" # Differentiated S3 path

# --- Multi-GPU Execution Logic --- 
echo "Starting multi-GPU training plan execution (Seed: $SEED_VAL)..."
echo "Debug: About to determine number of GPUs..."

# 1. Determine number of GPUs
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n 1)
echo "Debug: Raw NUM_GPUS_DETECTED: '$NUM_GPUS_DETECTED'"

if ! [[ "$NUM_GPUS_DETECTED" =~ ^[0-9]+$ ]]; then # This line might have been reverted by git pull, ensure it's the robust one
    echo "Warning: nvidia-smi failed or returned non-numeric/empty value ('$NUM_GPUS_DETECTED'). Assuming 1 GPU slot."
    NUM_GPUS=1
elif [[ "$NUM_GPUS_DETECTED" -eq 0 ]]; then
    echo "Warning: nvidia-smi reported 0 GPUs. Assuming 1 GPU slot (may run on CPU if no GPU 0 found by PyTorch)."
    NUM_GPUS=1
else
    NUM_GPUS=$NUM_GPUS_DETECTED
fi
echo "Debug: Determined NUM_GPUS: $NUM_GPUS"
echo "Will attempt to use $NUM_GPUS GPU slot(s) for parallel execution."

# Create output directory for script logs if it doesn't exist
SCRIPT_LOG_DIR="training_output/script_logs"
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
                  # Add WANDB_ENTITY if it's set and not empty
                  # Example: if [ -n "$WANDB_ENTITY" ]; then CMD+=("--wandb_entity" "$WANDB_ENTITY"); fi
                  "--run_suffix" "$RUN_SUFFIX" 
                  "--output_dir" "$LOCAL_TRAINING_OUTPUT_DIR" # Base output for train.py, it will create k-specific subdirs
                  "--upload_results_to_s3"
                  "--s3_results_bucket" "$S3_RESULTS_BUCKET"
                  "--s3_results_prefix" "$S3_RESULTS_PREFIX"
                )
                # Uncomment and fill if WANDB_ENTITY is used:
                # if [ -n "$WANDB_ENTITY" ]; then
                #   CMD+=("--wandb_entity" "$WANDB_ENTITY")
                # fi

                echo "Executing on GPU $gpu_id: ${CMD[@]}"
                echo "Script log for this job: $JOB_LOG_FILE"
                
                # Launch in background
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
echo "Individual `train.py` outputs and logs are in subdirectories under training_output/"
echo "--------------------------------------------------"

exit 0 