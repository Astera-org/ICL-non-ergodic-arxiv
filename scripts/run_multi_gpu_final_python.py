#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import shutil

# --- Configuration (Mirrors settings from the .sh script) ---
# --- Core Hyperparameters ---
MODEL_NAME_OR_PATH = "EleutherAI/pythia-70m-deduped"
BATCH_SIZE = 16
LEARNING_RATE = 0.0003
LR_SCHEDULE_TYPE = "constant_with_warmup"
NUM_WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 32
SEQUENCE_LENGTH = 256
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
PRECISION = "fp16"
MAX_EPOCHS_HARD_LIMIT = 5000  # Passed as --epochs
STEPS_PER_EVAL_EPOCH = 100
CHECKPOINT_INTERVAL = 2000      # Passed as --checkpoint_interval_steps
MAX_CHECKPOINTS = 12            # Passed as --max_step_checkpoints
GEOM_BETA = 0.95
MAX_LOSS_CKPTS = 0

# --- Early Stopping and ReduceLROnPlateau ---
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_DELTA = 0.0
REDUCE_LR_FACTOR = 1.0
REDUCE_LR_PATIENCE = 10
MIN_LR = 1e-6
NUM_DATALOADER_WORKERS = 2 # Passed as --num_workers

# --- W&B Configuration ---
WANDB_PROJECT = "icl-non-ergodic-arxiv"
WANDB_ENTITY = os.getenv("WANDB_ENTITY") # Optional, can be None

# --- Local Output Directory Configuration ---
LOCAL_TRAINING_OUTPUT_DIR_ROOT = Path("/data/users/adam/checkpoints")

# --- Experiment Grid ---
K_VALUES_TO_ITERATE = [1, 3, 5, 8, 11]
SEED_VALUES_TO_ITERATE = [0, 1, 2, 3]
RUN_SUFFIX_BASE = "final_python_orch" # Distinguishes runs from this script

# --- S3 Configuration ---
S3_RESULTS_BUCKET = "obelisk-simplex"
S3_RESULTS_PREFIX_ROOT = "non-ergodic-arxiv/training_runs"

# --- Other Fixed Args for train.py ---
NUM_BEST_EMA_VAL_CHECKPOINTS = 5
EVAL_DATASET_MULTIPLIER = 3

# Global lock for print statements to avoid interleaved output from threads
print_lock = threading.Lock()

def log_print(message: str):
    with print_lock:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def get_git_metadata() -> Dict[str, Any]:
    metadata = {"notes": "Metadata collected by Python orchestrator"}
    try:
        if shutil.which("git") and subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, check=False).stdout.strip() == "true":
            metadata["repository_root"] = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
            metadata["remote_url"] = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True, check=True).stdout.strip()
            metadata["current_branch"] = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            metadata["commit_hash"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            metadata["commit_date"] = subprocess.run(["git", "log", "-1", "--format=%cd"], capture_output=True, text=True, check=True).stdout.strip()
            metadata["commit_message"] = subprocess.run(["git", "log", "-1", "--format=%s"], capture_output=True, text=True, check=True).stdout.strip()
            
            diff_proc = subprocess.run(["git", "diff", "--quiet"], check=False)
            diff_staged_proc = subprocess.run(["git", "diff", "--staged", "--quiet"], check=False)
            if diff_proc.returncode == 0 and diff_staged_proc.returncode == 0:
                metadata["working_directory_status"] = "Clean"
            else:
                metadata["working_directory_status"] = "Dirty"
                metadata["diff"] = subprocess.run(["git", "diff"], capture_output=True, text=True, check=True).stdout.strip()
        else:
            metadata["git_status"] = "Not a git repository or git command not available"
    except Exception as e:
        log_print(f"Error gathering git metadata: {e}")
        metadata["git_error"] = str(e)
    return metadata

def save_overall_metadata(overall_run_dir: Path, experiment_timestamp: str, git_meta: Dict, script_params: Dict):
    metadata_dir = overall_run_dir / "metadata_overall"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / f"run_metadata_{experiment_timestamp}.json"
    
    full_metadata = {
        "experiment_timestamp": experiment_timestamp,
        "K_VALUES_TO_ITERATE": K_VALUES_TO_ITERATE,
        "SEED_VALUES_TO_ITERATE": SEED_VALUES_TO_ITERATE,
        "git_metadata": git_meta,
        "script_parameters": script_params,
        "wandb_config": {"WANDB_PROJECT": WANDB_PROJECT, "WANDB_ENTITY": WANDB_ENTITY},
        "s3_config": {"S3_RESULTS_BUCKET": S3_RESULTS_BUCKET, "S3_RESULTS_PREFIX_ROOT": S3_RESULTS_PREFIX_ROOT}
    }
    with open(metadata_file, 'w') as f:
        json.dump(full_metadata, f, indent=4)
    log_print(f"Overall metadata saved to {metadata_file}")
    return metadata_file

def get_num_gpus() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        # Take the first line of the output, strip whitespace, then convert to int
        first_line = result.stdout.strip().split('\n')[0]
        return int(first_line)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
        log_print(f"nvidia-smi not found or failed, or parsing failed. Defaulting to 1 GPU. Error: {e}")
        return 1

def run_single_train_job(job_config: Dict[str, Any], gpu_id: int, overall_run_dir: Path, script_log_dir: Path, s3_overall_prefix: str) -> Tuple[Dict[str, Any], bool]:
    k_val = job_config["k"]
    seed_val = job_config["seed"]
    
    formatted_lr = f"{LEARNING_RATE:.0e}".replace("e+0", "e").replace("e-0", "e-")
    job_specific_run_suffix = f"k{k_val}_s{seed_val}_lr{formatted_lr}_bs{BATCH_SIZE}_{RUN_SUFFIX_BASE}_{job_config['experiment_timestamp']}"
    specific_s3_prefix = f"{s3_overall_prefix}/{job_specific_run_suffix}"

    log_filename = f"run_k{k_val}_s{seed_val}_gpu{gpu_id}.log"
    log_filepath = script_log_dir / log_filename

    # Determine path to train.py relative to this script
    # Assumes this script is in 'scripts/' and train.py is in the project root (parent of 'scripts/')
    train_script_path = Path(__file__).parent.parent / "train.py"

    cmd = [
        sys.executable, str(train_script_path), # Use explicit path to train.py
        "--model_name_or_path", str(MODEL_NAME_OR_PATH),
        "--k", str(k_val),
        "--seed", str(seed_val),
        "--batch_size", str(BATCH_SIZE),
        "--sequence_length", str(SEQUENCE_LENGTH),
        "--learning_rate", str(LEARNING_RATE),
        "--lr_scheduler_type", str(LR_SCHEDULE_TYPE),
        "--num_warmup_steps", str(NUM_WARMUP_STEPS),
        "--weight_decay", str(WEIGHT_DECAY),
        "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
        "--adam_beta1", str(ADAM_BETA1),
        "--adam_beta2", str(ADAM_BETA2),
        "--adam_epsilon", str(ADAM_EPSILON),
        "--max_grad_norm", str(MAX_GRAD_NORM),
        "--precision", str(PRECISION),
        "--epochs", str(MAX_EPOCHS_HARD_LIMIT),
        "--steps_per_eval_epoch", str(STEPS_PER_EVAL_EPOCH),
        "--checkpoint_interval_steps", str(CHECKPOINT_INTERVAL),
        "--max_step_checkpoints", str(MAX_CHECKPOINTS),
        "--max_loss_ckpts", str(MAX_LOSS_CKPTS),
        "--geom_beta", str(GEOM_BETA),
        "--early_stopping_patience", str(EARLY_STOPPING_PATIENCE),
        "--early_stopping_delta", str(EARLY_STOPPING_DELTA),
        "--reduce_lr_factor", str(REDUCE_LR_FACTOR),
        "--reduce_lr_patience", str(REDUCE_LR_PATIENCE),
        "--min_lr", str(MIN_LR),
        "--num_workers", str(NUM_DATALOADER_WORKERS),
        "--output_dir", str(overall_run_dir), # train.py will create subdirectory based on run_suffix
        "--run_suffix", job_specific_run_suffix,
        "--wandb_project", WANDB_PROJECT,
        "--wandb_run_name", job_specific_run_suffix,
        "--upload_results_to_s3",
        "--s3_bucket", S3_RESULTS_BUCKET,
        "--s3_prefix", specific_s3_prefix,
        "--delete_local_checkpoints_after_s3_upload",
        "--num_best_ema_val_checkpoints", str(NUM_BEST_EMA_VAL_CHECKPOINTS),
        "--eval_dataset_multiplier", str(EVAL_DATASET_MULTIPLIER),
    ]
    if WANDB_ENTITY:
        cmd.extend(["--wandb_entity", WANDB_ENTITY])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    job_display_name = f"K={k_val}, Seed={seed_val} on GPU {gpu_id}"
    log_print(f"LAUNCHING: {job_display_name}. Log: {log_filepath}")
    # log_print(f"Full command: {' '.join(cmd)}") # Optional: very verbose

    start_time = time.time()
    success = False
    try:
        with open(log_filepath, 'w') as lf:
            process = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT) # Redirect stderr to stdout
            process.wait() # Wait for the process to complete
        
        duration = time.time() - start_time
        if process.returncode == 0:
            log_print(f"SUCCESS ({duration:.2f}s): {job_display_name}. Output log: {log_filepath}")
            success = True
        else:
            log_print(f"ERROR (code {process.returncode}, {duration:.2f}s): {job_display_name}. Output log: {log_filepath}")
    except Exception as e:
        duration = time.time() - start_time
        log_print(f"EXCEPTION ({duration:.2f}s) running {job_display_name}: {e}. Log: {log_filepath}")
        
    return job_config, success


def upload_to_s3(local_path: Path, s3_bucket: str, s3_prefix: str, is_recursive: bool = False):
    cmd = ["aws", "s3", "cp"]
    if is_recursive:
        cmd.append("--recursive")
    cmd.extend([str(local_path), f"s3://{s3_bucket}/{s3_prefix}"])
    
    try:
        log_print(f"Uploading {'directory' if is_recursive else 'file'} {local_path} to s3://{s3_bucket}/{s3_prefix}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log_print(f"Successfully uploaded {local_path} to s3://{s3_bucket}/{s3_prefix}")
    except subprocess.CalledProcessError as e:
        log_print(f"Failed to upload {local_path} to S3. Error: {e.stderr}")
    except FileNotFoundError:
        log_print(f"AWS CLI not found. Skipping S3 upload for {local_path}.")


def main():
    # --- Initial Setup ---
    experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    overall_run_dir = LOCAL_TRAINING_OUTPUT_DIR_ROOT / f"run_{experiment_timestamp}_{RUN_SUFFIX_BASE}"
    overall_run_dir.mkdir(parents=True, exist_ok=True)
    
    s3_overall_prefix = f"{S3_RESULTS_PREFIX_ROOT}/run_{experiment_timestamp}_{RUN_SUFFIX_BASE}"
    
    script_log_dir = overall_run_dir / "orchestrator_job_logs" # Logs for each train.py call
    script_log_dir.mkdir(parents=True, exist_ok=True)

    log_print(f"Starting Python orchestration script. Timestamp: {experiment_timestamp}")
    log_print(f"Overall local output directory: {overall_run_dir}")
    log_print(f"Overall S3 prefix: s3://{S3_RESULTS_BUCKET}/{s3_overall_prefix}")
    log_print(f"Individual train.py call logs will be in: {script_log_dir}")

    # --- Save Metadata ---
    # Ensure shutil is imported for which()
    git_meta = get_git_metadata()
    
    # Gather current script parameters for metadata
    script_params = {param: globals()[param] for param in [
        "MODEL_NAME_OR_PATH", "BATCH_SIZE", "LEARNING_RATE", "LR_SCHEDULE_TYPE", 
        "NUM_WARMUP_STEPS", "WEIGHT_DECAY", "GRADIENT_ACCUMULATION_STEPS", 
        "SEQUENCE_LENGTH", "ADAM_BETA1", "ADAM_BETA2", "ADAM_EPSILON", 
        "MAX_GRAD_NORM", "PRECISION", "MAX_EPOCHS_HARD_LIMIT", "STEPS_PER_EVAL_EPOCH",
        "CHECKPOINT_INTERVAL", "MAX_CHECKPOINTS", "GEOM_BETA", "MAX_LOSS_CKPTS",
        "EARLY_STOPPING_PATIENCE", "EARLY_STOPPING_DELTA", "REDUCE_LR_FACTOR",
        "REDUCE_LR_PATIENCE", "MIN_LR", "NUM_DATALOADER_WORKERS", 
        "NUM_BEST_EMA_VAL_CHECKPOINTS", "EVAL_DATASET_MULTIPLIER"
    ]}
    metadata_file_path = save_overall_metadata(overall_run_dir, experiment_timestamp, git_meta, script_params)


    # --- Prepare Jobs ---
    all_jobs_configs = []
    for k_val in K_VALUES_TO_ITERATE:
        for seed_val in SEED_VALUES_TO_ITERATE:
            all_jobs_configs.append({
                "k": k_val, 
                "seed": seed_val, 
                "experiment_timestamp": experiment_timestamp # Pass timestamp for consistent run_suffix
            })
    
    num_gpus = get_num_gpus()
    if num_gpus == 0:
        log_print("ERROR: No GPUs detected or nvidia-smi failed. Exiting.")
        return 1
    log_print(f"Detected {num_gpus} GPU(s). Will run up to {num_gpus} jobs in parallel.")

    # --- Execute Jobs ---
    completed_jobs = 0
    failed_jobs = 0
    total_jobs_to_run = len(all_jobs_configs)

    # GPU ID manager
    available_gpus = list(range(num_gpus))
    gpu_assignment_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        job_idx = 0
        
        # This loop submits jobs as GPUs become available (or initially)
        # A more robust approach might use a queue of jobs and workers pulling from it.
        # For now, let's manage GPU IDs directly.
        
        # Simplified GPU management: assign GPU IDs round-robin to futures
        # For true parallel execution up to num_gpus, the executor handles it.
        # We just need to ensure CUDA_VISIBLE_DEVICES is set appropriately.
        
        for i, job_conf in enumerate(all_jobs_configs):
            gpu_to_use = i % num_gpus # Simple round-robin for CUDA_VISIBLE_DEVICES
            futures.append(executor.submit(run_single_train_job, job_conf, gpu_to_use, overall_run_dir, script_log_dir, s3_overall_prefix))

        for future in as_completed(futures):
            job_config_returned, success = future.result() # Can get job_config if needed
            if success:
                completed_jobs += 1
            else:
                failed_jobs += 1
            log_print(f"PROGRESS: {completed_jobs} successful, {failed_jobs} failed out of {total_jobs_to_run} jobs.")

    log_print("--------------------------------------------------")
    log_print(f"All {total_jobs_to_run} training jobs have been processed.")
    log_print(f"Successful: {completed_jobs}, Failed: {failed_jobs}")
    log_print(f"Check individual job logs in: {script_log_dir}")
    log_print(f"Check train.py outputs in subdirectories of: {overall_run_dir}")
    log_print(f"Check S3 at s3://{S3_RESULTS_BUCKET}/{s3_overall_prefix}")
    log_print("--------------------------------------------------")

    # --- Upload Overall Artifacts ---
    overall_artifacts_s3_prefix = f"{s3_overall_prefix}/_overall_run_artifacts_python_orchestrator"
    log_print(f"Uploading overall run artifacts to s3://{S3_RESULTS_BUCKET}/{overall_artifacts_s3_prefix}")
    
    upload_to_s3(overall_run_dir / "metadata_overall", S3_RESULTS_BUCKET, f"{overall_artifacts_s3_prefix}/metadata_overall/", is_recursive=True)
    if script_log_dir.exists() and any(script_log_dir.iterdir()):
        upload_to_s3(script_log_dir, S3_RESULTS_BUCKET, f"{overall_artifacts_s3_prefix}/orchestrator_job_logs/", is_recursive=True)
    
    log_print("Python orchestration script finished.")
    return 0

if __name__ == "__main__":
    # Example of how to add command-line arguments to this script if needed later
    # parser = argparse.ArgumentParser(description="Python orchestrator for multi-GPU training.")
    # parser.add_argument("--custom_arg", type=str, help="An example custom argument.")
    # args = parser.parse_args()
    # if args.custom_arg:
    #    log_print(f"Custom argument provided: {args.custom_arg}")
    
    # Ensure Python 3.6+ for f-strings and other features, Pathlib
    if sys.version_info < (3, 6):
        sys.stderr.write("Python 3.6 or higher is required to run this script.\n")
        sys.exit(1)
        
    sys.exit(main()) 