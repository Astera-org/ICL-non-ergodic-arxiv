#!/bin/bash
# Description: Runs the full training plan based on EXPERIMENT_PLAN.md.
# Assumes the FULL preprocessed dataset exists locally (e.g., in ./preprocessed_arxiv/).
# Ensure you have enough compute resources (GPU recommended) and time.

# Source .env file if it exists
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration based on EXPERIMENT_PLAN.md Section 4 --- 
MODEL_NAME_OR_PATH="EleutherAI/pythia-70m-deduped" # Model config for random init
BATCH_SIZE=256           # Global batch size
LEARNING_RATE=5e-5       # Peak learning rate (Increased due to gradient accum)
LR_SCHEDULE_TYPE="cosine" # Cosine scheduler
NUM_WARMUP_STEPS=2000    # Warm-up steps (in optimizer steps)
WEIGHT_DECAY=0.1         # Weight decay
GRADIENT_ACCUMULATION_STEPS=8 # Accumulate over 8 micro-batches
# EPOCHS=12              # Deprecated by token budget / max_steps
TOKEN_BUDGET=2600000000  # Approx 100k steps * 256 batch * 101 tokens/seq

CHECKPOINT_INTERVAL=10000 # Save checkpoint every N steps
MAX_CHECKPOINTS=12        # Max checkpoints to keep

# W&B Configuration
WANDB_PROJECT="icl-non-ergodic-arxiv" # Default project name
# WANDB_ENTITY="your_entity"     # Optional: Your W&B username or team

# Experiment Grid (from EXPERIMENT_PLAN.md Section 3)
K_VALUES=(1 2 4 8 11)
# SEEDS=(0 1 2) # Removed - Only running seed 0

# --- Execution --- 
echo "Starting full training plan execution (SEED=0 ONLY)..."

# Ensure dataset exists (basic check)
if [ ! -f "preprocessed_arxiv/tokens.bin" ]; then
    echo "Error: Preprocessed data not found in ./preprocessed_arxiv/." 
    echo "Please run data preparation first (e.g., scripts/create_full_dataset_s3.sh and then download, or create locally)."
    exit 1
fi

# Ensure logged into W&B (optional, script will proceed without if disabled or login fails)
# Consider adding `wandb login --relogin` here if needed.

for k_val in "${K_VALUES[@]}"; do
  # for seed_val in "${SEEDS[@]}"; do # Removed SEED loop
    seed_val=0 # Hardcode seed to 0
    RUN_SUFFIX="full_plan_seed0" # Optional suffix for clarity
    echo "--------------------------------------------------"
    echo "Launching Training: K=$k_val, Seed=$seed_val"
    echo "--------------------------------------------------"
    
    # Construct the command
    CMD=(
      "python" "train.py" \
      "--model_name_or_path" "$MODEL_NAME_OR_PATH" \
      "--k" "$k_val" \
      "--seed" "$seed_val" \
      "--batch_size" "$BATCH_SIZE" \
      "--learning_rate" "$LEARNING_RATE" \
      "--lr_scheduler_type" "$LR_SCHEDULE_TYPE" \
      "--num_warmup_steps" "$NUM_WARMUP_STEPS" \
      "--weight_decay" "$WEIGHT_DECAY" \
      "--gradient_accumulation_steps" "$GRADIENT_ACCUMULATION_STEPS" \
      "--token_budget" "$TOKEN_BUDGET" \
      "--epochs" "1000" \
      "--checkpoint_interval_steps" "$CHECKPOINT_INTERVAL" \
      "--max_step_checkpoints" "$MAX_CHECKPOINTS" \
      "--wandb_project" "$WANDB_PROJECT" \
      # Uncomment and set if needed: "--wandb_entity" "$WANDB_ENTITY" \
      "--run_suffix" "$RUN_SUFFIX" \
      # Add S3 upload arguments
      "--upload_results_to_s3" \
      "--s3_results_bucket" "obelisk-simplex" \
      "--s3_results_prefix" "non-ergodic-arxiv/training_runs"
      # Add --force_cpu if needed, otherwise defaults to GPU if available
    )
    
    # Print the command before running
    echo "Running command: ${CMD[@]}"
    
    # Execute the command
    "${CMD[@]}"
    
    echo "--------------------------------------------------"
    echo "Finished Training: K=$k_val, Seed=$seed_val"
    echo "--------------------------------------------------"
    sleep 2 # Small delay between runs

# done # End SEEDS loop # Removed SEED loop end
done # End K_VALUES loop

echo "Full training plan execution finished (SEED=0 ONLY)." 