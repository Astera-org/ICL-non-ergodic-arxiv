#!/bin/bash
# Description: Runs the final training plan for multiple random seeds and K values,
# distributing runs across multiple available GPUs.
# Assumes the FULL preprocessed dataset exists locally (e.g., in ./preprocessed_arxiv/).
# Ensure you have enough compute resources (GPU recommended) and time.

# Source .env file if it exists
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.

# --- Generate timestamp for this experiment run ---\nEXPERIMENT_TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
echo "Starting final experiment run with timestamp: $EXPERIMENT_TIMESTAMP"

# --- Configuration (adapted from run_multi_gpu_training_plan.sh) ---
MODEL_NAME_OR_PATH="EleutherAI/pythia-70m-deduped"
BATCH_SIZE=16
LEARNING_RATE="0.0003"
# LR_SCHEDULE_TYPE="cosine" # Intentionally kept commented
NUM_WARMUP_STEPS=200
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=32
SEQUENCE_LENGTH=256
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPSILON=1e-8 # Default AdamW epsilon
MAX_GRAD_NORM=1.0
PRECISION="fp16"
MAX_EPOCHS_HARD_LIMIT=5000 # Max number of evaluation epochs
STEPS_PER_EVAL_EPOCH=100  # Number of optimizer steps per evaluation epoch
CHECKPOINT_INTERVAL=2000 # Optimizer steps between periodic checkpoints
MAX_CHECKPOINTS=12        # Max periodic checkpoints to keep
GEOM_ALPHA=0.95
GEOM_BETA=0.95
MAX_LOSS_CKPTS=0          # Max geometric (loss-based) checkpoints to keep (0 for unlimited)

# Early Stopping and ReduceLROnPlateau
EARLY_STOPPING_PATIENCE=30 # Eval epochs
EARLY_STOPPING_DELTA=0.0
REDUCE_LR_FACTOR=1.0       # Set to 1.0 to effectively disable ReduceLROnPlateau
REDUCE_LR_PATIENCE=10      # Eval epochs (effectively ignored due to REDUCE_LR_FACTOR=1.0)
MIN_LR=1e-6
NUM_DATALOADER_WORKERS=2

# W&B Configuration
WANDB_PROJECT="icl-non-ergodic-arxiv"
# WANDB_ENTITY="" # Optional: Your W&B username or team. Uncomment and set if needed.

# Local Output Directory Configuration
LOCAL_TRAINING_OUTPUT_DIR_ROOT="/data/users/adam/checkpoints/final_runs" # Define the root for final runs
LOCAL_TRAINING_OUTPUT_DIR="${LOCAL_TRAINING_OUTPUT_DIR_ROOT}/run_${EXPERIMENT_TIMESTAMP}"
mkdir -p "$LOCAL_TRAINING_OUTPUT_DIR"
echo "Local output directory for this run: $LOCAL_TRAINING_OUTPUT_DIR"

# Experiment Grid
K_VALUES_TO_ITERATE=(1 3 5 8 11) # Define K values to iterate over
SEED_VALUES_TO_ITERATE=(1 2 3)   # Define SEED values to iterate over
RUN_SUFFIX_BASE="final_multi_gpu"  # Base suffix for this final multi-GPU execution

# S3 Configuration
S3_RESULTS_BUCKET="obelisk-simplex"
S3_RESULTS_PREFIX_ROOT="non-ergodic-arxiv/training_runs_final_multi_gpu" # Differentiated S3 path for final runs
S3_RESULTS_PREFIX="${S3_RESULTS_PREFIX_ROOT}/run_${EXPERIMENT_TIMESTAMP}"
echo "S3 results will be uploaded to: s3://${S3_RESULTS_BUCKET}/${S3_RESULTS_PREFIX}"

# --- Save Git Repository Metadata for the entire run ---
echo "Gathering git repository metadata for the entire final run..."
METADATA_DIR="${LOCAL_TRAINING_OUTPUT_DIR}/metadata_overall"
mkdir -p "$METADATA_DIR"
METADATA_FILE="${METADATA_DIR}/git_metadata_final_run_${EXPERIMENT_TIMESTAMP}.txt"

echo "# Git Repository Metadata (Final Run - ${EXPERIMENT_TIMESTAMP})" > "$METADATA_FILE"
echo "# Generated at: $(date)" >> "$METADATA_FILE"
echo "# K_VALUES_TO_ITERATE: (${K_VALUES_TO_ITERATE[@]})" >> "$METADATA_FILE"
echo "# SEED_VALUES_TO_ITERATE: (${SEED_VALUES_TO_ITERATE[@]})" >> "$METADATA_FILE"
echo "" >> "$METADATA_FILE"
# (Git info capturing logic - reusing from run_multi_gpu_training_plan.sh)
echo "## Git Information" >> "$METADATA_FILE"
echo "Repository Root: $(pwd)" >> "$METADATA_FILE"
if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null; then
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
# (W&B and script config logging - reusing and adapting)
if [ -n "$WANDB_ENTITY" ]; then echo "WANDB_ENTITY: $WANDB_ENTITY" >> "$METADATA_FILE"; else echo "WANDB_ENTITY: Not specified" >> "$METADATA_FILE"; fi
if [ -n "$WANDB_API_KEY" ]; then echo "WANDB_API_KEY: [REDACTED]" >> "$METADATA_FILE"; else echo "WANDB_API_KEY: Not set" >> "$METADATA_FILE"; fi
echo "" >> "$METADATA_FILE"; echo "## Script Hyperparameters (Common)" >> "$METADATA_FILE"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH" >> "$METADATA_FILE"
echo "BATCH_SIZE: $BATCH_SIZE" >> "$METADATA_FILE"
echo "LEARNING_RATE: $LEARNING_RATE" >> "$METADATA_FILE"
echo "NUM_WARMUP_STEPS: $NUM_WARMUP_STEPS" >> "$METADATA_FILE"
# ... (add other common hyperparams)
echo "MAX_EPOCHS_HARD_LIMIT: $MAX_EPOCHS_HARD_LIMIT" >> "$METADATA_FILE"
echo "STEPS_PER_EVAL_EPOCH: $STEPS_PER_EVAL_EPOCH" >> "$METADATA_FILE"
echo "REDUCE_LR_FACTOR: $REDUCE_LR_FACTOR (1.0 means plateau LR reduction is off)" >> "$METADATA_FILE"
echo "S3_RESULTS_BUCKET: $S3_RESULTS_BUCKET" >> "$METADATA_FILE"
echo "S3_RESULTS_PREFIX_ROOT: $S3_RESULTS_PREFIX_ROOT" >> "$METADATA_FILE"
echo "Overall metadata saved to $METADATA_FILE"

# --- Multi-GPU Execution Logic ---
echo "Starting FINAL multi-GPU training plan execution..."

set +e
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n 1)
set -e

if ! [[ "$NUM_GPUS_DETECTED" =~ ^[0-9]+$ ]]; then
    echo "Warning: nvidia-smi failed. Assuming 1 GPU slot."
    NUM_GPUS=1
elif [[ "$NUM_GPUS_DETECTED" -eq 0 ]]; then
    echo "Warning: nvidia-smi reported 0 GPUs. Assuming 1 GPU slot."
    NUM_GPUS=1
else
    NUM_GPUS=$NUM_GPUS_DETECTED
fi
echo "Will use $NUM_GPUS GPU slot(s) for parallel execution."

SCRIPT_LOG_DIR="${LOCAL_TRAINING_OUTPUT_DIR}/script_logs_per_run"
mkdir -p "$SCRIPT_LOG_DIR"
echo "Individual run script logs will be in $SCRIPT_LOG_DIR"

# --- Build the full list of (K, SEED) combinations ---
JOBS_TO_RUN=()
for K_VAL in "${K_VALUES_TO_ITERATE[@]}"; do
  for SEED_VAL in "${SEED_VALUES_TO_ITERATE[@]}"; do
    JOBS_TO_RUN+=("$K_VAL,$SEED_VAL") # Store as "K,SEED" string
  done
done
TOTAL_JOBS=${#JOBS_TO_RUN[@]}
echo "Total individual training jobs to run: $TOTAL_JOBS"

# Associative array to store PIDs of jobs on each GPU. Key: gpu_id, Value: pid
declare -A pids_on_gpu
# Associative array to store K,SEED string for the job on a GPU. Key: gpu_id, Value: "K,SEED"
declare -A job_details_on_gpu

for (( i=0; i<$NUM_GPUS; i++ )); do
    pids_on_gpu[$i]=0
    job_details_on_gpu[$i]=""
done

completed_job_count=0

while [[ $completed_job_count -lt $TOTAL_JOBS ]]; do
    for gpu_id in $(seq 0 $(($NUM_GPUS - 1)) ); do
        current_pid=${pids_on_gpu[$gpu_id]}

        if [[ $current_pid -eq 0 ]] || ! kill -0 "$current_pid" 2>/dev/null; then
            if [[ $current_pid -ne 0 ]]; then
                echo "INFO: Job for ${job_details_on_gpu[$gpu_id]} (PID $current_pid) on GPU $gpu_id has completed."
                ((completed_job_count++))
                echo "INFO: Completed $completed_job_count out of $TOTAL_JOBS total jobs."
                pids_on_gpu[$gpu_id]=0
                job_details_on_gpu[$gpu_id]=""
            fi
            
            if [[ ${#JOBS_TO_RUN[@]} -gt 0 ]]; then
                job_to_launch_str="${JOBS_TO_RUN[0]}"
                JOBS_TO_RUN=("${JOBS_TO_RUN[@]:1}")

                # Parse K and SEED from the string
                IFS=',' read -r K_TO_LAUNCH SEED_TO_LAUNCH <<< "$job_to_launch_str"

                echo "--------------------------------------------------"
                echo "LAUNCHING: Training for K=$K_TO_LAUNCH, SEED=$SEED_TO_LAUNCH on GPU $gpu_id"
                echo "--------------------------------------------------"
                
                JOB_LOG_FILE="${SCRIPT_LOG_DIR}/run_k${K_TO_LAUNCH}_seed${SEED_TO_LAUNCH}_gpu${gpu_id}.log"
                
                # Construct W&B run name and train.py output subdirectory name
                # Ensure LEARNING_RATE is consistently formatted (e.g., no leading/trailing zeros if treated as string later)
                FORMATTED_LR=$(printf "%.0e" "$LEARNING_RATE" | sed 's/e+*0*/e/' | sed 's/e-0*/e-/') # e.g., 3e-4
                
                # Ensure unique run name for W&B and directory for train.py
                # Include K, SEED, LR, BS, and a base suffix + overall experiment timestamp
                TRAIN_PY_RUN_NAME="k${K_TO_LAUNCH}_s${SEED_TO_LAUNCH}_lr${FORMATTED_LR}_bs${BATCH_SIZE}_${RUN_SUFFIX_BASE}_${EXPERIMENT_TIMESTAMP}"
                OUTPUT_SUBDIR_FOR_TRAIN_PY="${LOCAL_TRAINING_OUTPUT_DIR}/${TRAIN_PY_RUN_NAME}"
                mkdir -p "$OUTPUT_SUBDIR_FOR_TRAIN_PY" # train.py will also create this, but good practice

                # Path for the specific run's copy of the overall metadata
                RUN_METADATA_TARGET_PATH="${OUTPUT_SUBDIR_FOR_TRAIN_PY}/git_metadata_overall_run.txt"
                cp "$METADATA_FILE" "$RUN_METADATA_TARGET_PATH"
                echo "Overall metadata copied to: $RUN_METADATA_TARGET_PATH for this specific run."

                EMA_VAL_LOSS_ALPHA=0.5 # This seems to be a leftover or specific to a version of train.py, check if still needed

                # Ensure environment variables for W&B are available if set
                WANDB_CMD_ENV=""
                if [ -n "$WANDB_ENTITY" ]; then WANDB_CMD_ENV+="WANDB_ENTITY=$WANDB_ENTITY "; fi
                # WANDB_API_KEY is usually handled by wandb login or system config, not passed directly in cmd normally
                
                # Launch the training script in the background
                (
                  echo "Starting train.py for K=$K_TO_LAUNCH, SEED=$SEED_TO_LAUNCH on GPU $gpu_id"
                  echo "Output logs will be in: $JOB_LOG_FILE"
                  echo "Model checkpoints and run artifacts in: $OUTPUT_SUBDIR_FOR_TRAIN_PY"
                  echo "Full command:"
                  # Print the command for easier debugging
                  set -x # Echo commands
                  CUDA_VISIBLE_DEVICES=$gpu_id ${WANDB_CMD_ENV}python train.py \
                    --model_name_or_path "$MODEL_NAME_OR_PATH" \
                    --k "$K_TO_LAUNCH" \
                    --seed "$SEED_TO_LAUNCH" \
                    --batch_size "$BATCH_SIZE" \
                    --sequence_length "$SEQUENCE_LENGTH" \
                    --learning_rate "$LEARNING_RATE" \
                    --num_warmup_steps "$NUM_WARMUP_STEPS" \
                    --weight_decay "$WEIGHT_DECAY" \
                    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
                    --adam_beta1 "$ADAM_BETA1" \
                    --adam_beta2 "$ADAM_BETA2" \
                    --adam_epsilon "$ADAM_EPSILON" \
                    --max_grad_norm "$MAX_GRAD_NORM" \
                    --precision "$PRECISION" \
                    --epochs "$MAX_EPOCHS_HARD_LIMIT" \
                    --steps_per_eval_epoch "$STEPS_PER_EVAL_EPOCH" \
                    --checkpoint_interval_steps "$CHECKPOINT_INTERVAL" \
                    --max_step_checkpoints "$MAX_CHECKPOINTS" \
                    --max_loss_ckpts "$MAX_LOSS_CKPTS" \
                    --geom_alpha "$GEOM_ALPHA" \
                    --geom_beta "$GEOM_BETA" \
                    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
                    --early_stopping_delta "$EARLY_STOPPING_DELTA" \
                    --reduce_lr_factor "$REDUCE_LR_FACTOR" \
                    --reduce_lr_patience "$REDUCE_LR_PATIENCE" \
                    --min_lr "$MIN_LR" \
                    --num_workers "$NUM_DATALOADER_WORKERS" \
                    --output_dir "$OUTPUT_SUBDIR_FOR_TRAIN_PY" \
                    --wandb_project "$WANDB_PROJECT" \
                    --wandb_run_name "$TRAIN_PY_RUN_NAME" \
                    --upload_results_to_s3 \
                    --s3_bucket "$S3_RESULTS_BUCKET" \
                    --s3_prefix "${S3_RESULTS_PREFIX}/${TRAIN_PY_RUN_NAME}" \
                    --delete_local_checkpoints_after_s3_upload \
                    --num_best_ema_val_checkpoints 5 \
                    # --force_cpu # Ensure this is not set for GPU runs
                    # --disable_wandb # Ensure this is not set
                  set +x
                  echo "train.py for K=$K_TO_LAUNCH, SEED=$SEED_TO_LAUNCH on GPU $gpu_id finished."
                ) > "$JOB_LOG_FILE" 2>&1 &
                
                # Store PID and K,SEED info
                pids_on_gpu[$gpu_id]=$!
                job_details_on_gpu[$gpu_id]="$K_TO_LAUNCH,$SEED_TO_LAUNCH"
                
                echo "INFO: Launched K=$K_TO_LAUNCH, SEED=$SEED_TO_LAUNCH with PID ${pids_on_gpu[$gpu_id]} on GPU $gpu_id. Log: $JOB_LOG_FILE"
                
                # Optional: Short delay to allow script to start and potentially avoid resource contention during init
                sleep 5 
            fi
        fi
    done
    # Wait for a bit before checking GPU statuses again if not all jobs are done
    if [[ $completed_job_count -lt $TOTAL_JOBS ]]; then
        sleep 10 # Check every 10 seconds
    fi
done

echo "--------------------------------------------------"
echo "All $TOTAL_JOBS training jobs have been processed."
echo "Final experiment run $EXPERIMENT_TIMESTAMP completed."
echo "Check logs in $SCRIPT_LOG_DIR and $LOCAL_TRAINING_OUTPUT_DIR"
echo "Check S3 at s3://${S3_RESULTS_BUCKET}/${S3_RESULTS_PREFIX}"
echo "--------------------------------------------------"

# Optionally, upload the overall metadata and script logs to S3 as well
OVERALL_ARTIFACTS_S3_PREFIX="${S3_RESULTS_PREFIX}/_overall_run_artifacts"
echo "Uploading overall run artifacts (metadata, script logs) to s3://${S3_RESULTS_BUCKET}/${OVERALL_ARTIFACTS_S3_PREFIX}"
aws s3 cp --recursive "$METADATA_DIR" "s3://${S3_RESULTS_BUCKET}/${OVERALL_ARTIFACTS_S3_PREFIX}/metadata_overall/"
aws s3 cp --recursive "$SCRIPT_LOG_DIR" "s3://${S3_RESULTS_BUCKET}/${OVERALL_ARTIFACTS_S3_PREFIX}/script_logs_per_run/"
echo "Overall artifacts upload complete." 