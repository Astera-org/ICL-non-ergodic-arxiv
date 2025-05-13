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

# Step 1: List training run folders from S3
def list_training_runs(bucket_name, experiment_path):
    """List all training run folders in the S3 bucket."""
    print(f"Step 1: Listing training runs in {bucket_name}/{experiment_path}")
    
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    training_run_dirs = []
    for result in paginator.paginate(Bucket=bucket_name, Prefix=experiment_path, Delimiter='/'):
        for common_prefix in result.get('CommonPrefixes', []):
            training_run_dirs.append(common_prefix.get('Prefix'))
    
    print(f"Found {len(training_run_dirs)} training runs:")
    for folder in training_run_dirs:
        print(folder)
    
    return training_run_dirs

# Step 2: Find the folders within a specific training run
def list_run_folders(bucket_name, training_run_dir):
    """List folders within a specific training run."""
    print(f"\nStep 2: Listing folders in {training_run_dir}")
    
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=training_run_dir, Delimiter='/')
    folders = response.get('CommonPrefixes', [])
    
    print(f"Found {len(folders)} folders:")
    for folder in folders:
        print(folder.get('Prefix'))
    
    return [folder.get('Prefix') for folder in folders]

# Step 3: Find the training args file for a run
def find_training_args(bucket_name, training_run_dir):
    """Find the training args JSON file for a specific run."""
    print(f"\nStep 3: Looking for training args file in {training_run_dir}")
    
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=training_run_dir)
    
    training_args_path = None
    for file in response.get('Contents', []):
        file_key = file.get('Key')
        if file_key.endswith('_args.json'):
            training_args_path = f"s3://{bucket_name}/{file_key}"
            break
    
    if training_args_path:
        print(f"Training args file found: {training_args_path}")
    else:
        print("No training args file found in this directory.")
    
    return training_args_path

# Step 4: Find the model checkpoint with the lowest loss
def find_best_checkpoint(bucket_name, training_run_dir):
    """Find the model checkpoint with the lowest validation loss."""
    print(f"\nStep 4: Finding model checkpoint with lowest loss")
    
    s3_client = boto3.client('s3')
    loss_checkpoints_prefix = training_run_dir + 'loss_checkpoints/'
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
            print(f"Could not parse loss value from folder: {folder_name}")
    
    if lowest_loss_folder:
        print(f"Lowest loss folder: {lowest_loss_folder} (loss: {lowest_loss})")
        
        # Get model and config files
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
        
        print(f"Model path: {model_path}")
        print(f"Config path: {config_path}")
        print(f"Generation config path: {generation_config_path}")
        
        return {
            'model_path': model_path,
            'config_path': config_path,
            'generation_config_path': generation_config_path
        }
    else:
        print("No valid loss checkpoint folders found.")
        return None

# Step 5: Download and parse the model config file
def download_and_parse_config(bucket_name, config_path):
    """Download and parse the model config file."""
    print(f"\nStep 5: Downloading and parsing config file")
    
    if not config_path:
        print("No config path provided")
        return None
        
    s3_client = boto3.client('s3')
    s3_key = config_path.replace(f"s3://{bucket_name}/", "")
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        print(f"Downloading config from {config_path}...")
        s3_client.download_file(bucket_name, s3_key, tmp_path)
        
        with open(tmp_path, 'r') as f:
            config_data = json.load(f)
        
        print("Config JSON contents:")
        print(json.dumps(config_data, indent=2))
        
        return config_data
    except Exception as e:
        print(f"Error downloading or parsing config: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Step 6: Download and load the model
def download_and_load_model(bucket_name, model_files):
    """Download model files and load the model."""
    print(f"\nStep 6: Downloading and loading the model")
    
    model_path = model_files.get('model_path')
    config_path = model_files.get('config_path')
    generation_config_path = model_files.get('generation_config_path')
    
    if not model_path or not config_path:
        print("Missing model or config paths.")
        return None
    
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
    
    # Download generation config if available
    if generation_config_path:
        generation_config_s3_key = generation_config_path.replace(f"s3://{bucket_name}/", "")
        local_generation_config_path = os.path.join(model_dir, "generation_config.json")
        print(f"Downloading generation config...")
        s3_client.download_file(bucket_name, generation_config_s3_key, local_generation_config_path)
    
    # Load the model
    try:
        print("Loading model...")
        config = AutoConfig.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            device_map="auto"  # Use GPU if available, else CPU
        )
        print(f"Model loaded successfully: {model.__class__.__name__}")
        print(f"Model parameters: {model.num_parameters():,}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Step 7: Download and parse training args
def download_and_parse_args(bucket_name, training_args_path):
    """Download and parse the training args file."""
    print(f"\nStep 7: Downloading and parsing training args")
    
    if not training_args_path:
        print("No training args path provided")
        return None
        
    s3_client = boto3.client('s3')
    args_s3_key = training_args_path.replace(f"s3://{bucket_name}/", "")
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        print(f"Downloading training args...")
        s3_client.download_file(bucket_name, args_s3_key, tmp_path)
        
        with open(tmp_path, 'r') as f:
            args_data = json.load(f)
        
        print("Training args JSON contents:")
        print(json.dumps(args_data, indent=2))
        
        # Print key info
        print("\nKey training parameters:")
        print(f"Base model: {args_data.get('model_name_or_path', 'N/A')}")
        print(f"Learning rate: {args_data.get('learning_rate', 'N/A')}")
        print(f"Batch size: {args_data.get('batch_size', 'N/A')}")
        
        return args_data
    except Exception as e:
        print(f"Error downloading or parsing training args: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Step 8: Create validation dataset and dataloader
def create_validation_dataloader(args_data):
    """Create a validation dataset and dataloader using the args from training."""
    print(f"\nStep 8: Creating validation dataset and dataloader")
    
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
    
    # Get parameters from args or use defaults
    k = args_data.get('k', 3) if args_data else 3
    seed = args_data.get('seed', 42) if args_data else 42
    sequence_length = args_data.get('sequence_length', 256) if args_data else 256
    batch_size = args_data.get('batch_size', 8) if args_data else 8
    
    # Scale the multiplier by 11/k to keep total samples approximately constant across different k values
    base_multiplier = 100
    scaling_factor = 11 / k
    eval_dataset_multiplier = int(base_multiplier * scaling_factor)
    print(f"Using scaled multiplier: {eval_dataset_multiplier} (base: {base_multiplier} × scaling: {scaling_factor:.2f})")
    
    preprocessed_data_path = Path(args_data.get('preprocessed_data_dir', DEFAULT_PREPROCESSED_DIR)) if args_data else Path(DEFAULT_PREPROCESSED_DIR)
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Select categories
    selected_categories = select_categories(ALL_CATEGORIES, k, seed)
    print(f"Selected {k} categories for validation (seed {seed}): {selected_categories}")
    
    # Create dataset
    try:
        val_dataset = RandomWindowDataset(
            preprocessed_dir=preprocessed_data_path,
            split="validation",
            target_categories=selected_categories,
            sequence_length=sequence_length,
            eval_multiplier=eval_dataset_multiplier
        )
        
        # Create dataloader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True
        ) if len(val_dataset) > 0 else None
        
        print(f"Validation dataset size: {len(val_dataset)} examples")
        if val_dataloader:
            print(f"Validation dataloader created with {len(val_dataloader)} batches")
            print(f"Total samples to process: {len(val_dataset) * batch_size}")
        else:
            print("Validation dataloader is None (empty dataset)")
        
        return val_dataloader
    except Exception as e:
        print(f"Error creating validation dataset/dataloader: {e}")
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

# Step 9: Get a batch from the dataloader
def get_first_batch(val_dataloader):
    """Get the first batch from the validation dataloader."""
    print(f"\nStep 9: Getting the first batch from the validation dataloader")
    
    if val_dataloader is None:
        print("Validation dataloader is None")
        return None
    
    first_batch = next(iter(val_dataloader))
    print(f"First batch shape: {first_batch.shape}")
    print(f"First batch device: {first_batch.device}")
    print(f"First 5 tokens of first sequence: {first_batch[0, :5].tolist()}")
    
    return first_batch

# Step 10: Compute loss for a single batch
def compute_batch_loss(model, batch, device='cpu'):
    """Compute loss for a single batch."""
    batch = batch.to(device)
    model.eval()
    
    with torch.no_grad():
        # Method 1: Get loss directly from model by providing labels
        outputs = model(input_ids=batch, labels=batch)
        model_loss = outputs.loss
        
        # Method 2: Compute loss manually from logits
        outputs_no_labels = model(batch)
        logits = outputs_no_labels.logits
        
        # Shift logits and labels for loss calculation
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        # Flatten
        loss_fct = torch.nn.CrossEntropyLoss()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_labels = shift_labels.view(-1)
        
        # Compute loss
        manual_loss = loss_fct(flat_shift_logits, flat_shift_labels)
    
    return model_loss.item(), manual_loss.item()

# Step 11: Evaluate on the entire validation dataset
def evaluate_dataset(model, dataloader, device='cpu'):
    """Compute average loss over the entire dataset with a progress bar."""
    print(f"\nStep 11: Evaluating on the entire validation dataset")
    
    if dataloader is None:
        print("Validation dataloader is None")
        return None, None
    
    model.eval()
    total_model_loss = 0.0
    total_manual_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating validation set", leave=True):
            batch_model_loss, batch_manual_loss = compute_batch_loss(model, batch, device)
            total_model_loss += batch_model_loss
            total_manual_loss += batch_manual_loss
            num_batches += 1
    
    # Compute average losses
    avg_model_loss = total_model_loss / num_batches if num_batches > 0 else float('inf')
    avg_manual_loss = total_manual_loss / num_batches if num_batches > 0 else float('inf')
    
    # Calculate perplexity (standard metric for language models)
    model_perplexity = torch.exp(torch.tensor(avg_model_loss)).item()
    manual_perplexity = torch.exp(torch.tensor(avg_manual_loss)).item()
    
    print(f"Average loss over {num_batches} batches:")
    print(f"Model computed loss: {avg_model_loss:.6f}")
    print(f"Manually computed loss: {avg_manual_loss:.6f}")
    print(f"Model perplexity: {model_perplexity:.6f}")
    print(f"Manual perplexity: {manual_perplexity:.6f}")
    
    return avg_model_loss, avg_manual_loss

# Step 12: Compute per-position loss and plot it
def compute_and_plot_position_loss(model, dataloader, sequence_length, device='cpu', run_name="unknown_run"):
    """
    Compute loss at each position in the context window across the entire dataset
    and plot the results.
    """
    print(f"\nStep 12: Computing and plotting loss per context window position")
    
    if dataloader is None:
        print("Validation dataloader is None")
        return None
    
    model.eval()
    
    # Initialize accumulators for per-position losses
    # Start from position 1 since position 0 has no previous context to predict from
    per_position_loss_sum = torch.zeros(sequence_length, device=device)
    per_position_counts = torch.zeros(sequence_length, device=device, dtype=torch.long)
    
    # For more detailed statistics, collect all losses for each position
    all_position_losses = [[] for _ in range(sequence_length)]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing per-position loss", leave=True):
            batch = batch.to(device)
            
            # Get outputs from model
            outputs = model(batch)
            logits = outputs.logits
            
            # Shift logits and labels for causal language modeling
            # logits[:, :-1, :] predicts tokens at positions 1 to sequence_length-1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            
            # We need to keep the per-position losses separate, so use reduction='none'
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_labels = shift_labels.view(-1)
            
            # Compute per-token losses [batch_size * (sequence_length-1)]
            token_losses = loss_fct(flat_shift_logits, flat_shift_labels)
            
            # Reshape to [batch_size, sequence_length-1]
            token_losses = token_losses.view(batch.size(0), -1)
            
            # For each position in the prediction window (1 to sequence_length-1)
            # accumulate losses and counts
            actual_prediction_len = shift_logits.size(1)  # sequence_length - 1
            
            # Sum losses across batch for each position
            # Add to positions 1 to actual_prediction_len in the accumulators
            per_position_loss_sum[1:actual_prediction_len+1] += token_losses.sum(dim=0)
            per_position_counts[1:actual_prediction_len+1] += batch.size(0)
            
            # Also collect individual losses for more statistics
            for pos in range(1, actual_prediction_len+1):
                pos_losses = token_losses[:, pos-1].detach().cpu().numpy().tolist()
                all_position_losses[pos].extend(pos_losses)
    
    # Compute average loss at each position
    per_position_avg_loss = torch.zeros_like(per_position_loss_sum)
    for pos in range(1, sequence_length):
        if per_position_counts[pos] > 0:
            per_position_avg_loss[pos] = per_position_loss_sum[pos] / per_position_counts[pos]
        else:
            per_position_avg_loss[pos] = float('nan')
    
    # Convert to numpy for plotting
    positions = np.arange(sequence_length)
    losses = per_position_avg_loss.cpu().numpy()
    
    # Exclude position 0 from the plot (no prediction for the first token)
    valid_positions = positions[1:]
    valid_losses = losses[1:]
    
    # Calculate statistics
    valid_mean = np.nanmean(valid_losses)
    valid_min = np.nanmin(valid_losses)
    valid_max = np.nanmax(valid_losses)
    
    # Calculate additional statistics for each position
    position_stats = {}
    for pos in range(1, sequence_length):
        if len(all_position_losses[pos]) > 0:
            pos_losses = np.array(all_position_losses[pos])
            position_stats[pos] = {
                'mean': np.mean(pos_losses),
                'median': np.median(pos_losses),
                'std': np.std(pos_losses),
                'var': np.var(pos_losses),
                'sem': np.std(pos_losses) / np.sqrt(len(pos_losses)),
                'min': np.min(pos_losses),
                'max': np.max(pos_losses),
                'count': len(pos_losses)
            }
    
    print(f"Loss statistics across positions:")
    print(f"  Mean: {valid_mean:.4f}")
    print(f"  Min:  {valid_min:.4f} (position {valid_positions[np.nanargmin(valid_losses)]})")
    print(f"  Max:  {valid_max:.4f} (position {valid_positions[np.nanargmax(valid_losses)]})")
    
    # Create figure for main plot
    plt.figure(figsize=(14, 8))
    
    # Plot raw position loss
    plt.plot(valid_positions, valid_losses, '-', color='blue', alpha=0.7, linewidth=1.5)
    
    # Add markers for every 25 positions
    marker_positions = np.arange(25, sequence_length, 25)
    marker_indices = np.array([pos - 1 for pos in marker_positions if pos < len(valid_positions)])
    marker_values = [valid_losses[idx] for idx in marker_indices]
    
    plt.plot(valid_positions[marker_indices], marker_values, 'o', color='blue', alpha=0.8)
    
    # Remove moving average calculation and plotting
    
    # Add horizontal line at the global average loss
    plt.axhline(y=valid_mean, color='green', linestyle='--', alpha=0.7, 
               label=f'Global Mean: {valid_mean:.4f}')
    
    # Add grid, labels, and title
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Position in Context Window', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title(f'Loss by Position in Context Window - {run_name}', fontsize=14)
    
    # Add statistics annotation
    stats_text = (f"Statistics:\n"
                 f"Mean: {valid_mean:.4f}\n"
                 f"Min: {valid_min:.4f} (pos {valid_positions[np.nanargmin(valid_losses)]})\n"
                 f"Max: {valid_max:.4f} (pos {valid_positions[np.nanargmax(valid_losses)]})\n"
                 f"Mean Perplexity: {np.exp(valid_mean):.2f}")
    
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                 verticalalignment='top', fontsize=10)
    
    # Add perplexity reference on a second y-axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.set_ylim(np.exp(ax1.get_ylim()[0]), np.exp(ax1.get_ylim()[1]))
    ax2.set_ylabel('Perplexity (exp(Loss))', fontsize=12)
    
    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              fancybox=True, shadow=True, ncol=3)
    
    plt.tight_layout()
    
    # Save the main plot
    plot_path = f'position_loss_plot_{run_name}.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    print(f"Main plot saved to {plot_path}")
    
    # Create a secondary plot - showing first 50 positions
    if len(valid_positions) > 50:
        plt.figure(figsize=(12, 6))
        plt.plot(valid_positions[:50], valid_losses[:50], 'o-', 
                color='blue', alpha=0.7, markersize=4)
        
        # Remove smoothed line plotting for zoomed view
        
        plt.axhline(y=valid_mean, color='green', linestyle='--', alpha=0.7, 
                   label=f'Global Mean: {valid_mean:.4f}')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Position in Context Window', fontsize=12)
        plt.ylabel('Average Loss', fontsize=12)
        plt.title(f'Loss by Position in Context Window (First 50 Positions) - {run_name}', fontsize=14)
        plt.legend()
        plt.tight_layout()
        
        # Save this zoomed-in plot
        zoomed_plot_path = f'position_loss_plot_first50_{run_name}.png'
        plt.savefig(zoomed_plot_path, dpi=150)
        print(f"Zoomed plot saved to {zoomed_plot_path}")
    
    # Also save the data for future analysis
    data_to_save = {
        'run_name': run_name,
        'positions': valid_positions.tolist(),
        'losses': valid_losses.tolist(),
        'mean_loss': valid_mean,
        'min_loss': valid_min,
        'max_loss': valid_max,
        'min_position': valid_positions[np.nanargmin(valid_losses)],
        'max_position': valid_positions[np.nanargmax(valid_losses)],
        'position_stats': position_stats
    }
    results_dir = 'position_loss_results'
    os.makedirs(results_dir, exist_ok=True)
    np.save(f'{results_dir}/position_loss_data_{run_name}.npy', data_to_save)
    
    # Also save as JSON for easier inspection
    with open(f'{results_dir}/position_loss_data_{run_name}.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        data_for_json = {
            'run_name': run_name,
            'positions': valid_positions.tolist(),
            'losses': valid_losses.tolist(),
            'mean_loss': float(valid_mean),
            'min_loss': float(valid_min),
            'max_loss': float(valid_max),
            'min_position': int(valid_positions[np.nanargmin(valid_losses)]),
            'max_position': int(valid_positions[np.nanargmax(valid_losses)]),
            'position_stats': {
                str(pos): {
                    k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                    for k, v in stats.items()
                }
                for pos, stats in position_stats.items()
            }
        }
        json.dump(data_for_json, f, indent=2)
    
    print(f"Loss data saved to {results_dir}/position_loss_data_{run_name}.npy and .json")
    
    return data_to_save

# Main function that ties everything together
def main():
    """Main function to run all steps in sequence."""
    # Constants
    BUCKET_NAME = 'obelisk-simplex'
    EXPERIMENT_PATH = 'non-ergodic-arxiv/training_runs/run_2025-05-11_05-33-44_final_python_orch/'
    
    # Get the appropriate device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Step 1: List training runs
    training_runs = list_training_runs(BUCKET_NAME, EXPERIMENT_PATH)
    
    if not training_runs:
        print("No training runs found. Exiting.")
        return
    
    # Create a directory for all results
    results_dir = 'position_loss_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a summary file for all runs
    summary_file = f'{results_dir}/all_runs_summary.csv'
    with open(summary_file, 'w') as f:
        f.write("run_name,mean_loss,min_loss,min_position,max_loss,max_position,mean_perplexity\n")
    
    # Loop through all training runs (or a subset for testing)
    for run_idx, run_path in enumerate(training_runs):
        try:
            # Extract run name for files
            run_name = run_path.strip('/').split('/')[-1]
            print(f"\n{'-'*80}")
            print(f"Processing run {run_idx+1}/{len(training_runs)}: {run_name}")
            print(f"{'-'*80}")
            
            # Step 2: List folders in this run
            list_run_folders(BUCKET_NAME, run_path)
            
            # Step 3: Find training args file
            training_args_path = find_training_args(BUCKET_NAME, run_path)
            
            # Step 4: Find the best model checkpoint
            model_files = find_best_checkpoint(BUCKET_NAME, run_path)
            
            if not model_files:
                print(f"No model checkpoint found for {run_name}. Skipping.")
                continue
            
            # Step 5: Download and parse config
            download_and_parse_config(BUCKET_NAME, model_files.get('config_path'))
            
            # Step 6: Download and load model
            model = download_and_load_model(BUCKET_NAME, model_files)
            
            if model is None:
                print(f"Failed to load model for {run_name}. Skipping.")
                continue
            
            # Step 7: Download and parse training args
            args_data = download_and_parse_args(BUCKET_NAME, training_args_path)
            
            # Step 8: Create validation dataloader
            val_dataloader = create_validation_dataloader(args_data)
            
            if val_dataloader is None:
                print(f"Failed to create validation dataloader for {run_name}. Skipping.")
                continue
            
            # Step 9: Get a batch from the dataloader (quick check)
            batch = get_first_batch(val_dataloader)
            
            if batch is None:
                print(f"Failed to get a batch for {run_name}. Skipping.")
                continue
            
            # Step 10: Run a batch through the model to check loss
            print("\nStep 10: Computing loss from model outputs")
            batch = batch.to(device)
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                # Method 1: Get loss directly from model
                outputs = model(input_ids=batch, labels=batch)
                model_loss = outputs.loss
                print(f"Loss computed by model: {model_loss.item()}")
                
                # Method 2: Compute loss manually
                outputs_no_labels = model(batch)
                logits = outputs_no_labels.logits
                
                # Shift logits and labels for loss calculation
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                
                # Flatten
                loss_fct = torch.nn.CrossEntropyLoss()
                flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_shift_labels = shift_labels.view(-1)
                
                # Compute loss
                manual_loss = loss_fct(flat_shift_logits, flat_shift_labels)
                print(f"Loss computed manually: {manual_loss.item()}")
                
                # Calculate perplexity
                perplexity = torch.exp(manual_loss)
                print(f"Perplexity: {perplexity.item()}")
            
            # Step 11: Evaluate on the entire validation dataset
            avg_model_loss, avg_manual_loss = evaluate_dataset(model, val_dataloader, device)
            
            # Step 12: Compute per-position loss and plot it
            sequence_length = args_data.get('sequence_length', 256) if args_data else 256
            position_data = compute_and_plot_position_loss(model, val_dataloader, sequence_length, device, run_name)
            
            # Append to summary file
            with open(summary_file, 'a') as f:
                f.write(f"{run_name},{position_data['mean_loss']},{position_data['min_loss']},{position_data['min_position']}," +
                        f"{position_data['max_loss']},{position_data['max_position']},{np.exp(position_data['mean_loss'])}\n")
            
            # Free up GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
        except Exception as e:
            print(f"Error processing run {run_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nAll runs processed. Summary file saved to {summary_file}")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main() 