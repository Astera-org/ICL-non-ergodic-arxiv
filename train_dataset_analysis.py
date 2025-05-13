import boto3
import json
import os
import tempfile
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import re

# Add project root to sys.path to allow importing RandomWindowDataset
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if MPS (Apple Silicon GPU) is available
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def download_args_file(bucket_name, args_path):
    """Download and parse the training args file."""
    s3_client = boto3.client('s3')
    args_s3_key = args_path.replace(f"s3://{bucket_name}/", "")
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        print(f"Downloading args file...")
        s3_client.download_file(bucket_name, args_s3_key, tmp_path)
        
        with open(tmp_path, 'r') as f:
            args_data = json.load(f)
        
        return args_data
    except Exception as e:
        print(f"Error downloading or parsing args: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def find_model_files(bucket_name, run_path):
    """Find the model checkpoint files in a run directory."""
    s3_client = boto3.client('s3')
    loss_checkpoints_prefix = run_path + 'loss_checkpoints/'
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=loss_checkpoints_prefix, Delimiter='/')
    checkpoint_folders = response.get('CommonPrefixes', [])
    
    lowest_loss = float('inf')
    lowest_loss_folder = None
    
    # Parse folder names to extract loss values
    for folder in checkpoint_folders:
        folder_prefix = folder.get('Prefix')
        folder_name = folder_prefix.split('/')[-2]  # Get the folder name without trailing slash
        try:
            # Format is loss_X.XX_stepYYYYYY
            loss_str = folder_name.split('_')[1]
            loss_value = float(loss_str)
            
            if loss_value < lowest_loss:
                lowest_loss = loss_value
                lowest_loss_folder = folder_prefix
        except (IndexError, ValueError):
            pass
    
    if lowest_loss_folder:
        files_response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=lowest_loss_folder)
        model_path = None
        config_path = None
        generation_config_path = None
        
        for file in files_response.get('Contents', []):
            file_key = file.get('Key')
            if file_key.endswith('model.safetensors'):
                model_path = f"s3://{bucket_name}/{file_key}"
            elif file_key.endswith('generation_config.json'):
                generation_config_path = f"s3://{bucket_name}/{file_key}"
            elif file_key.endswith('config.json'):
                config_path = f"s3://{bucket_name}/{file_key}"
        
        return {
            'model_path': model_path,
            'config_path': config_path,
            'generation_config_path': generation_config_path,
            'loss': lowest_loss
        }
    
    return None

def download_and_load_model(bucket_name, model_files, run_name):
    """Download model files and load the model."""
    model_path = model_files.get('model_path')
    config_path = model_files.get('config_path')
    
    if not model_path or not config_path:
        return None
    
    # Use a single directory for all models, consistent with the original script
    model_dir = "downloaded_model"
    os.makedirs(model_dir, exist_ok=True)
    
    s3_client = boto3.client('s3')
    
    # Download model file
    model_s3_key = model_path.replace(f"s3://{bucket_name}/", "")
    local_model_path = os.path.join(model_dir, "model.safetensors")
    print(f"Downloading model...")
    s3_client.download_file(bucket_name, model_s3_key, local_model_path)
    
    # Download config file
    config_s3_key = config_path.replace(f"s3://{bucket_name}/", "")
    local_config_path = os.path.join(model_dir, "config.json")
    print(f"Downloading config...")
    s3_client.download_file(bucket_name, config_s3_key, local_config_path)
    
    # Check if there's a generation config path provided in the model_files
    generation_config_path = model_files.get('generation_config_path')
    if generation_config_path:
        generation_config_s3_key = generation_config_path.replace(f"s3://{bucket_name}/", "")
        local_generation_config_path = os.path.join(model_dir, "generation_config.json")
        print(f"Downloading generation config...")
        s3_client.download_file(bucket_name, generation_config_s3_key, local_generation_config_path)
    
    # Load the model
    try:
        print(f"Loading model for {run_name}...")
        config = AutoConfig.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            device_map="auto"  # Use GPU if available, else CPU
        )
        print(f"Model loaded successfully: {model.__class__.__name__}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_training_dataloader(seed, multiplier=1100):
    """Create a training dataloader for k=1 with the specified seed."""
    try:
        from random_window_dataset import RandomWindowDataset, DEFAULT_PREPROCESSED_DIR
    except ImportError:
        print("Could not import RandomWindowDataset. Make sure it's in the current directory or Python path.")
        return None
    
    # Define categories
    ALL_CATEGORIES = [
        "cs.CV", "cs.AI", "cs.SY", "cs.CE", "cs.PL",
        "cs.IT", "cs.DS", "cs.NE", "math.AC", "math.GR", "math.ST"
    ]
    ALL_CATEGORIES.sort()
    
    # Parameters
    k = 1  # Always use k=1 for consistency
    sequence_length = 256
    batch_size = 16
    eval_dataset_multiplier = multiplier  # Use the specified multiplier
    preprocessed_data_path = Path(DEFAULT_PREPROCESSED_DIR)
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Select categories
    selected_categories = select_categories(ALL_CATEGORIES, k, seed)
    print(f"Selected {k} categories for training (seed {seed}): {selected_categories}")
    
    # Create dataset
    try:
        train_dataset = RandomWindowDataset(
            preprocessed_dir=preprocessed_data_path,
            split="train",  # Use training split instead of validation
            target_categories=selected_categories,
            sequence_length=sequence_length,
            eval_multiplier=eval_dataset_multiplier
        )
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True
        ) if len(train_dataset) > 0 else None
        
        print(f"Training dataset size: {len(train_dataset)} examples")
        if train_dataloader:
            print(f"Training dataloader created with {len(train_dataloader)} batches")
            print(f"Total samples to process: {len(train_dataset) * batch_size}")
        
        return train_dataloader
    except Exception as e:
        print(f"Error creating training dataset/dataloader: {e}")
        return None

# Helper function to select categories
def select_categories(all_categories, k, seed):
    """Select k categories based on the seed."""
    if k < 1 or k > len(all_categories):
        raise ValueError(f"K must be between 1 and {len(all_categories)}, got {k}")
    
    sorted_cats = sorted(list(all_categories))
    rng = random.Random(seed)
    shuffled_cats = list(sorted_cats)
    rng.shuffle(shuffled_cats)
    selected = shuffled_cats[:k]
    return sorted(selected)

def compute_model_position_loss(model, dataloader, run_name, device='cpu', num_iterations=1000):
    """Compute loss at each position for a model."""
    print(f"\nComputing position loss for {run_name}")
    
    if model is None or dataloader is None:
        print("Model or dataloader is None")
        return None
    
    model.eval()
    sequence_length = 256  # Standard for our models
    
    # Initialize accumulators for per-position losses
    per_position_loss_sum = torch.zeros(sequence_length, device=device)
    per_position_counts = torch.zeros(sequence_length, device=device, dtype=torch.long)
    
    with torch.no_grad():
        # Iterate through the dataloader multiple times to get more samples
        for iteration in tqdm(range(num_iterations), desc=f"Computing loss for {run_name}", leave=True):
            for batch in dataloader:
                batch = batch.to(device)
                
                # Get outputs from model
                outputs = model(batch)
                logits = outputs.logits
                
                # Shift logits and labels for causal language modeling
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                
                # We need to keep the per-position losses separate
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_shift_labels = shift_labels.view(-1)
                
                # Compute per-token losses
                token_losses = loss_fct(flat_shift_logits, flat_shift_labels)
                
                # Reshape to [batch_size, sequence_length-1]
                token_losses = token_losses.view(batch.size(0), -1)
                
                # Accumulate losses for each position
                actual_prediction_len = shift_logits.size(1)  # sequence_length - 1
                
                # Sum losses across batch for each position
                # Add to positions 1 to actual_prediction_len in the accumulators
                per_position_loss_sum[1:actual_prediction_len+1] += token_losses.sum(dim=0)
                per_position_counts[1:actual_prediction_len+1] += batch.size(0)
    
    # Compute average loss at each position
    per_position_avg_loss = torch.zeros_like(per_position_loss_sum)
    for pos in range(1, sequence_length):
        if per_position_counts[pos] > 0:
            per_position_avg_loss[pos] = per_position_loss_sum[pos] / per_position_counts[pos]
        else:
            per_position_avg_loss[pos] = float('nan')
    
    # Convert to numpy
    positions = np.arange(sequence_length)
    losses = per_position_avg_loss.cpu().numpy()
    
    # Exclude position 0 (no prediction for the first token)
    valid_positions = positions[1:]
    valid_losses = losses[1:]
    
    mean_loss = np.nanmean(valid_losses)
    print(f"Mean loss for {run_name}: {mean_loss:.4f}")
    print(f"Total samples processed: {per_position_counts[1].item()} for each position")
    
    # Add additional statistics
    median_loss = np.nanmedian(valid_losses)
    std_loss = np.nanstd(valid_losses)
    sem_loss = std_loss / np.sqrt(np.sum(~np.isnan(valid_losses)))
    
    print(f"Median loss: {median_loss:.4f}")
    print(f"Loss standard deviation: {std_loss:.4f}")
    print(f"Standard error of the mean: {sem_loss:.4f}")
    
    return {
        'run_name': run_name,
        'positions': valid_positions,
        'losses': valid_losses,
        'mean_loss': mean_loss,
        'median_loss': median_loss,
        'std_loss': std_loss,
        'sem_loss': sem_loss,
        'sample_count': per_position_counts[1].item()
    }

def plot_seed_comparison(seed, results_by_k):
    """Create plots comparing different k values for a specific seed."""
    print(f"\nCreating comparison plots for seed {seed}")
    
    # Make sure we have results to plot
    if not results_by_k:
        print(f"No results to plot for seed {seed}")
        return
    
    # Create the plots directory
    plots_dir = f"train_seed_comparison_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create full plot for all positions
    plt.figure(figsize=(14, 8))
    
    # Use a consistent color scheme
    colors = {
        1: 'blue',
        3: 'green',
        5: 'red',
        8: 'purple',
        11: 'orange'
    }
    
    # Plot each k value
    for k, result in results_by_k.items():
        positions = result['positions']
        losses = result['losses']
        mean_loss = result['mean_loss']
        std_loss = result.get('std_loss', 0)
        median_loss = result.get('median_loss', 0)
        sem_loss = result.get('sem_loss', 0)
        sample_count = result.get('sample_count', 0)
        
        plt.plot(positions, losses, '-', color=colors.get(k, 'gray'), 
                 linewidth=1.5, label=f'k={k} (mean={mean_loss:.4f}, median={median_loss:.4f}, n={sample_count})')
    
    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Position in Context Window', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title(f'Training Loss by Position for Different k Values (Seed {seed})', fontsize=14)
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    full_plot_path = f'{plots_dir}/all_positions_seed{seed}.png'
    plt.savefig(full_plot_path, bbox_inches='tight', dpi=150)
    print(f"Full plot saved to {full_plot_path}")
    
    # Create zoomed plot for the first 50 positions
    plt.figure(figsize=(14, 8))
    
    # Plot each k value for the first 50 positions
    for k, result in results_by_k.items():
        positions = result['positions'][:50]
        losses = result['losses'][:50]
        mean_loss = np.nanmean(losses)
        std_loss = result.get('std_loss', 0)
        median_loss = result.get('median_loss', 0)
        sem_loss = result.get('sem_loss', 0)
        sample_count = result.get('sample_count', 0)
        
        plt.plot(positions, losses, '-', color=colors.get(k, 'gray'), 
                 linewidth=1.5, label=f'k={k} (mean={mean_loss:.4f}, median={median_loss:.4f}, n={sample_count})')
    
    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Position in Context Window', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title(f'Training Loss by Position for Different k Values (Seed {seed}, First 50 Positions)', fontsize=14)
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the zoomed plot
    zoomed_plot_path = f'{plots_dir}/first50_positions_seed{seed}.png'
    plt.savefig(zoomed_plot_path, bbox_inches='tight', dpi=150)
    print(f"Zoomed plot saved to {zoomed_plot_path}")
    
    # Also save the raw data
    save_data = {
        'seed': seed,
        'results_by_k': results_by_k
    }
    data_path = f'{plots_dir}/loss_data_seed{seed}.json'
    with open(data_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        data_for_json = {}
        for k, result in results_by_k.items():
            data_for_json[k] = {
                'run_name': result['run_name'],
                'positions': result['positions'].tolist() if isinstance(result['positions'], np.ndarray) else result['positions'],
                'losses': result['losses'].tolist() if isinstance(result['losses'], np.ndarray) else result['losses'],
                'mean_loss': float(result['mean_loss']),
                'median_loss': float(result.get('median_loss', 0)),
                'std_loss': float(result.get('std_loss', 0)),
                'sem_loss': float(result.get('sem_loss', 0)),
                'sample_count': int(result.get('sample_count', 0))
            }
        json.dump(data_for_json, f, indent=2)
    
    print(f"Data saved to {data_path}")

def main():
    # Constants
    BUCKET_NAME = 'obelisk-simplex'
    EXPERIMENT_PATH = 'non-ergodic-arxiv/training_runs/run_2025-05-11_05-33-44_final_python_orch/'
    
    # Get the appropriate device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Compile regex pattern for parsing run names
    run_pattern = re.compile(r'k(\d+)_s(\d+)_lr')
    
    # List all runs
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    run_paths = []
    for result in paginator.paginate(Bucket=BUCKET_NAME, Prefix=EXPERIMENT_PATH, Delimiter='/'):
        for common_prefix in result.get('CommonPrefixes', []):
            run_path = common_prefix.get('Prefix')
            match = run_pattern.search(run_path)
            if match:
                k = int(match.group(1))
                s = int(match.group(2))
                run_paths.append((run_path, k, s))
    
    # Group runs by seed
    runs_by_seed = {}
    for run_path, k, s in run_paths:
        if s not in runs_by_seed:
            runs_by_seed[s] = []
        runs_by_seed[s].append((run_path, k))
    
    # Process each seed
    for seed, runs in runs_by_seed.items():
        print(f"\n{'-'*80}")
        print(f"Processing seed {seed}")
        print(f"{'-'*80}")
        
        # Create dataloader for this seed
        print(f"Creating dataloader for k=1, seed={seed}")
        dataloader = create_training_dataloader(seed)
        
        if not dataloader:
            print(f"Failed to create dataloader for seed {seed}. Skipping.")
            continue
        
        # Process each run for this seed
        results_by_k = {}
        for run_path, k in sorted(runs, key=lambda x: x[1]):  # Sort by k
            try:
                # Extract run name
                run_name = run_path.strip('/').split('/')[-1]
                print(f"\nProcessing run: {run_name} (k={k}, seed={seed})")
                
                # Find model files
                model_files = find_model_files(BUCKET_NAME, run_path)
                if not model_files:
                    print(f"No model files found for {run_name}. Skipping.")
                    continue
                
                # Download and load model
                model = download_and_load_model(BUCKET_NAME, model_files, run_name)
                if not model:
                    print(f"Failed to load model for {run_name}. Skipping.")
                    continue
                
                # Compute per-position loss
                model = model.to(device)
                position_results = compute_model_position_loss(model, dataloader, run_name, device)
                
                if position_results:
                    results_by_k[k] = position_results
                
                # Free up GPU memory
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                
            except Exception as e:
                print(f"Error processing run {run_path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create comparison plots for this seed
        plot_seed_comparison(seed, results_by_k)
    
    print("\nAll seeds processed.")

if __name__ == "__main__":
    main() 