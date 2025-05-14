import subprocess
import itertools
import argparse
import os
import sys
from typing import List, Tuple, Dict, Optional
import time

# Ensure the script can find modules in the 'src' directory, assuming 'scripts' is a sibling of 'src'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

# Base categories available in the HDF5 files (as identified by inspect_hdf5_attrs.py)
DEFAULT_AVAILABLE_CATEGORIES = sorted(['cs.DS', 'cs.IT', 'cs.PL', 'math.GR', 'math.ST']) # Sorted for deterministic combinations

def format_categories_for_cli(categories: Tuple[str, ...]) -> str:
    """Formats a list of categories for Hydra CLI override, e.g., "['cat1','cat2']"."""
    items_str = ','.join([repr(str(cat)) for cat in categories])
    return f"[{items_str}]"

def generate_experiment_id(k: int, categories: Tuple[str, ...]) -> str:
    """Generates a descriptive experiment ID."""
    cat_short_names = "_".join(cat.replace('.', '') for cat in categories)
    return f"k{k}_{cat_short_names}"

def run_experiment_in_background(
    python_executable: str,
    gpu_id: int,
    k_value: int,
    active_categories: Tuple[str, ...],
    epochs: int,
    max_global_steps: int,
    logging_level: str,
    base_hydra_config_dir: str,
    output_log_dir: str,
    script_path: str = "src.main_hydra_app"
) -> Optional[subprocess.Popen]:
    """Constructs and runs a single experiment command in the background, assigning a GPU."""
    
    experiment_id = generate_experiment_id(k_value, active_categories)
    active_categories_cli = format_categories_for_cli(active_categories)

    # Ensure output log directory exists
    os.makedirs(output_log_dir, exist_ok=True)
    stdout_log_path = os.path.join(output_log_dir, f"{experiment_id}_stdout.log")
    stderr_log_path = os.path.join(output_log_dir, f"{experiment_id}_stderr.log")

    command = [
        python_executable,
        "-m", script_path,
        f"--config-dir={base_hydra_config_dir}",
        "experiment=default_experiment",
        f"experiment.k_value={k_value}",
        f"experiment.active_categories={active_categories_cli}",
        f"experiment.experiment_id={experiment_id}",
        f"training.epochs={epochs}",
        f"training.max_global_steps={max_global_steps}",
        f"logging.level={logging_level.upper()}",
        "training.perform_validation=True",
        "training.early_stopping_enabled=True" 
    ]

    print(f"\n--- Launching Experiment (GPU {gpu_id}): {experiment_id} ---")
    print(f"  K: {k_value}, Categories: {active_categories}")
    # print(f"  Full Command: {' '.join(command)}") # Can be very long
    print(f"  Stdout Log: {stdout_log_path}")
    print(f"  Stderr Log: {stderr_log_path}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        with open(stdout_log_path, 'wb') as stdout_file, open(stderr_log_path, 'wb') as stderr_file:
            process = subprocess.Popen(command, env=env, cwd=PROJECT_ROOT, stdout=stdout_file, stderr=stderr_file)
        print(f"  Launched {experiment_id} with PID: {process.pid} on GPU {gpu_id}")
        return process
    except Exception as e:
        print(f"  ERROR launching experiment {experiment_id} on GPU {gpu_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run a fixed set of 5 (or fewer) experiments in parallel on multiple GPUs.")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs for each training run."
    )
    parser.add_argument(
        "--max_global_steps", type=int, default=-1, 
        help="Max global steps for each training run (-1 for no limit, follows epochs)."
    )
    parser.add_argument(
        "--categories", nargs="+", default=DEFAULT_AVAILABLE_CATEGORIES,
        help="List of available categories to choose from for combinations. Order matters for selecting specific combinations."
    )
    parser.add_argument(
        "--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for the training script sub-processes."
    )
    parser.add_argument(
        "--python_exe", type=str, default=os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
        help="Path to the python executable in the virtual environment."
    )
    parser.add_argument(
        "--base_config_dir_name", type=str, default="configs",
        help="Name of the base Hydra config directory (relative to project root)."
    )
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Total number of GPUs available for cycling."
    )
    parser.add_argument(
        "--output_log_dir", type=str, default=os.path.join(PROJECT_ROOT, "experiment_logs"),
        help="Directory to store stdout/stderr logs for each experiment run."
    )

    args = parser.parse_args()

    if not os.path.exists(args.python_exe):
        print(f"ERROR: Python executable not found at {args.python_exe}")
        sys.exit(1)

    abs_base_hydra_config_dir = os.path.join(PROJECT_ROOT, args.base_config_dir_name)
    if not os.path.isdir(abs_base_hydra_config_dir):
        print(f"ERROR: Base Hydra config directory not found at {abs_base_hydra_config_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_log_dir, exist_ok=True)

    print(f"Starting parallel experiment script...")
    print(f"Python executable: {args.python_exe}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Max Global Steps per run: {args.max_global_steps}")
    print(f"Base categories for combinations: {args.categories}")
    print(f"Logging level for sub-processes: {args.logging_level}")
    print(f"Hydra config directory: {abs_base_hydra_config_dir}")
    print(f"Number of GPUs to cycle through: {args.num_gpus}")
    print(f"Output logs will be in: {args.output_log_dir}")

    experiments_to_run: List[Dict] = []
    # We want exactly one run for k=1, k=2, k=3, k=4, k=5
    # using the first combination of categories for each k.
    # Ensure categories are sorted for deterministic selection if the input order varies.
    sorted_categories = sorted(list(set(args.categories))) # Unique and sorted

    for k_val in range(1, 6): # k from 1 to 5
        if k_val > len(sorted_categories):
            print(f"Cannot form a combination for k={k_val} with only {len(sorted_categories)} unique categories. Stopping generation.")
            break
        # Get the first combination of k_val categories
        # itertools.combinations returns tuples
        first_combo_for_k = list(itertools.combinations(sorted_categories, k_val))[0]
        experiments_to_run.append({
            "k_value": k_val,
            "active_categories": first_combo_for_k
        })
    
    if len(experiments_to_run) > 5:
        experiments_to_run = experiments_to_run[:5] # Should not happen with current logic but as safeguard

    print(f"\nPlanning to run {len(experiments_to_run)} experiments:")
    for i, exp_params in enumerate(experiments_to_run):
        print(f"  {i+1}. k={exp_params['k_value']}, categories={exp_params['active_categories']}")

    processes: Dict[str, subprocess.Popen] = {}
    experiment_details: Dict[str, Dict] = {}

    for i, exp_params in enumerate(experiments_to_run):
        gpu_id_to_use = i % args.num_gpus # Cycle through available GPUs
        exp_id = generate_experiment_id(exp_params['k_value'], exp_params['active_categories'])
        experiment_details[exp_id] = exp_params

        process = run_experiment_in_background(
            python_executable=args.python_exe,
            gpu_id=gpu_id_to_use,
            k_value=exp_params['k_value'],
            active_categories=exp_params['active_categories'],
            epochs=args.epochs,
            max_global_steps=args.max_global_steps,
            logging_level=args.logging_level,
            base_hydra_config_dir=abs_base_hydra_config_dir,
            output_log_dir=args.output_log_dir
        )
        if process:
            processes[exp_id] = process
        else:
            print(f"Failed to launch experiment {exp_id}. It will not be monitored.")
        time.sleep(5) # Stagger launches slightly

    if not processes:
        print("No experiments were successfully launched.")
        sys.exit(1)

    print(f"\nAll {len(processes)} experiments launched. Monitoring for completion...")
    
    completed_count = 0
    failed_count = 0
    running_pids = {pid: exp_id for exp_id, p_obj in processes.items() for pid in [p_obj.pid] if p_obj.pid is not None}

    while len(running_pids) > 0:
        for exp_id, process_obj in list(processes.items()): # Iterate over a copy for modification
            if process_obj.pid not in running_pids: # Already handled
                continue
            
            return_code = process_obj.poll()
            if return_code is not None: # Process has finished
                if return_code == 0:
                    print(f"Experiment {exp_id} (PID {process_obj.pid}) completed successfully.")
                    completed_count += 1
                else:
                    print(f"Experiment {exp_id} (PID {process_obj.pid}) failed with return code {return_code}.")
                    print(f"  Check logs in {args.output_log_dir}/{exp_id}_stderror.log")
                    failed_count += 1
                del processes[exp_id]
                if process_obj.pid in running_pids:
                    del running_pids[process_obj.pid]
            # else: process is still running
        
        if len(running_pids) > 0:
            print(f"Waiting for {len(running_pids)} experiments to complete... PIDs: {list(running_pids.keys())}")
            time.sleep(30) # Wait before polling again

    print("\n--- Experiment Series Complete ---")
    print(f"Total experiments launched: {len(experiment_details)}")
    print(f"Successfully completed: {completed_count}")
    print(f"Failed experiments: {failed_count}")

if __name__ == "__main__":
    main() 