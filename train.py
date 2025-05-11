import argparse
import json
import logging
import os
import random
import shutil # For managing step checkpoints
import sys
import time # For timing steps
from pathlib import Path
from typing import List, Dict, Any, Optional # Added Optional here

import torch
import torch.optim # Import torch.optim
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_scheduler # Added AutoConfig
from tqdm import tqdm # For progress bars
import wandb # Added wandb
from dotenv import load_dotenv
import boto3 # Added boto3 again (might be needed for upload func)
from botocore.exceptions import ClientError # Added for S3 error handling
from torch.cuda.amp import GradScaler, autocast # For mixed precision
import functools # For partial in hook lambda

import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for environments without a display
import matplotlib.pyplot as plt

# Add project root to sys.path to allow importing RandomWindowDataset
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from random_window_dataset import RandomWindowDataset, DEFAULT_PREPROCESSED_DIR # noqa
# from fetch_arxiv import TOKENIZER_NAME # We'll define tokenizer name directly for now or get from args

# Define the 11 target categories from EXPERIMENT_PLAN.md
# These are the categories from which K will be selected.
ALL_CATEGORIES = [
    "cs.CV", "cs.AI", "cs.SY", "cs.CE", "cs.PL",
    "cs.IT", "cs.DS", "cs.NE", "math.AC", "math.GR", "math.ST"
]
ALL_CATEGORIES.sort() # Ensure canonical order

# Default tokenizer
DEFAULT_MODEL_NAME = "EleutherAI/pythia-70m-deduped"

# Load .env file as early as possible
load_dotenv()

def select_categories(all_categories: List[str], k: int, seed: int) -> List[str]:
    """
    Selects K categories deterministically based on a seed.
    Sorts the base list, shuffles a copy, then selects the first K.
    The selected list is also sorted for consistent run behavior.

    Args:
        all_categories: The full list of available category names.
        k: The number of categories to select.
        seed: The random seed to use for shuffling.

    Returns:
        A list containing the K selected category names.
    """
    if k < 1 or k > len(all_categories):
        raise ValueError(f"K must be between 1 and {len(all_categories)}, got {k}")

    sorted_cats = sorted(list(all_categories)) # Ensure base is sorted

    rng = random.Random(seed) # Use a separate Random instance for local, seeded shuffling
    shuffled_cats = list(sorted_cats) # Create a copy
    rng.shuffle(shuffled_cats)

    selected = shuffled_cats[:k]
    return sorted(selected) # Return sorted selected list


def setup_logging(output_dir: Path, run_name: str):
    """Sets up logging to file and console."""
    log_file = output_dir / f"{run_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging to {log_file}")

def save_args(args: argparse.Namespace, output_dir: Path, run_name: str):
    """Saves arguments to a JSON file, converting Path objects to strings."""
    args_path = output_dir / f"{run_name}_args.json"
    args_dict = vars(args).copy() # Make a copy to modify
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    logging.info(f"Saved arguments to {args_path}")

# --- GEOM CKPT SAVE ---
def save_geom_ckpt(model, step, loss_val, loss_ckpt_dir, geom_saved_list, args_ref, wandb_ref, logging_ref, shutil_ref, 
                   val_dataloader, device_ref, current_epoch_idx, evaluate_fn, 
                   geom_xs, geom_ys_list, geom_keys, output_dir_for_run_ref): # Pass plot data lists and output_dir_for_run
    ckpt_path = loss_ckpt_dir / f"loss_{loss_val:.2f}_step{step:06d}"
    model.save_pretrained(ckpt_path)
    geom_saved_list.append(ckpt_path)
    logging_ref.info(f"[GEOM] checkpoint @ step {step}  EMA={loss_val:.3f}")
    
    # Log the training EMA loss that triggered this checkpoint to W&B
    if not args_ref.disable_wandb and wandb_ref.run is not None:
        wandb_ref.log({
            "train/geom_ema_loss_at_ckpt": loss_val, 
            "train/geom_optimizer_step_at_ckpt": step    
        })

    # Evaluate on validation set if available and save per-token loss data
    if val_dataloader is not None:
        logging_ref.info(f"[GEOM] Evaluating on validation set at step {step} (epoch {current_epoch_idx+1}) due to geometric checkpoint, calculating per-token loss.")
        geom_overall_eval_loss, geom_per_token_eval_losses = evaluate_fn(model, val_dataloader, device_ref, current_epoch_idx, args_ref, calculate_per_token_loss=True)
        
        logging_ref.info(f"[GEOM] Overall validation loss at step {step}: {geom_overall_eval_loss:.4f}")
        
        # Log overall validation loss to W&B
        if not args_ref.disable_wandb and wandb_ref.run is not None:
            wandb_log_payload = {
                "eval/geom_val_loss_at_step": geom_overall_eval_loss,
                "eval/geom_val_loss_optimizer_step": step,
                "eval/geom_val_loss_epoch_context": current_epoch_idx + 1 
            }
            wandb_ref.log(wandb_log_payload) # Log scalar validation metrics

        # Save per-token loss data locally as JSON
        if geom_per_token_eval_losses is not None:
            logging_ref.info(f"[GEOM] Per-token validation losses calculated for step {step}.")
            geom_ys_list.append(geom_per_token_eval_losses) # Append the new Y series
            geom_keys.append(f"Step {step} | EMA {loss_val:.3f}")   # Append the key for this series
            
            plot_data_to_save = {
                "xs": geom_xs,
                "ys_list": geom_ys_list,
                "keys": geom_keys,
                "title": "In-Context Loss Profile (Geometric Ckpts)",
                "xname": "Token Position in Context Window"
            }
            json_output_path = output_dir_for_run_ref / "in_context_loss_profiles.json"
            try:
                with open(json_output_path, 'w') as f:
                    json.dump(plot_data_to_save, f, indent=4)
                logging_ref.info(f"[GEOM] Saved in-context loss profile data to {json_output_path}")
                
                # Also generate and save a plot image
                plot_image_path = output_dir_for_run_ref / "in_context_loss_profiles.png"
                try:
                    generate_loss_profile_plot(plot_data_to_save, plot_image_path, logging_ref)
                except Exception as e_plot_save:
                    logging_ref.error(f"[GEOM] Failed to generate/save loss profile plot image: {e_plot_save}")
                    
            except Exception as e_json_save:
                logging_ref.error(f"[GEOM] Failed to save in-context loss profile data to JSON: {e_json_save}")
        else:
            logging_ref.warning(f"[GEOM] Per-token validation losses were not returned for step {step}, cannot save profile data.")
            
    if args_ref.max_loss_ckpts > 0 and len(geom_saved_list) > args_ref.max_loss_ckpts:
        oldest = geom_saved_list.pop(0)
        shutil_ref.rmtree(oldest, ignore_errors=True)
        logging_ref.info(f"[GEOM] removed oldest {oldest}")
# -----------------------

# --- Function to generate and save the loss profile plot ---
def generate_loss_profile_plot(plot_data_dict, image_output_path, logging_ref):
    """Generates a plot from the loss profile data and saves it as an image."""
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        xs = plot_data_dict.get("xs")
        ys_list = plot_data_dict.get("ys_list", [])
        keys = plot_data_dict.get("keys", [])
        
        if not xs or not ys_list:
            logging_ref.warning("[PLOT] No data (xs or ys_list) to plot for loss profile.")
            plt.close(fig)
            return

        for i, ys in enumerate(ys_list):
            label = keys[i] if i < len(keys) else f"Series {i+1}"
            # Ensure ys is a flat list of numbers, not nested lists or other structures if not intended
            # Matplotlib expects y to be 1D array of the same new shape as x or 2D array with columns being plotted
            ax.plot(xs, ys, label=label, marker='.', linestyle='-') # Added marker for better visibility of points
            
        ax.set_title(plot_data_dict.get("title", "In-Context Loss Profile"))
        ax.set_xlabel(plot_data_dict.get("xname", "Token Position"))
        ax.set_ylabel("Average Loss")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize='small') # Add a legend
        
        plt.tight_layout()
        fig.savefig(image_output_path, dpi=150)
        plt.close(fig) # Close the figure to free memory
        logging_ref.info(f"[PLOT] Saved loss profile plot to {image_output_path}")
    except Exception as e:
        logging_ref.error(f"[PLOT] Error generating loss profile plot: {e}")
        if 'fig' in locals() and fig is not None: # Ensure fig exists before trying to close
            plt.close(fig)
# ----------------------------------------------------------

def upload_directory_to_s3(local_directory: Path, bucket: str, s3_prefix: str):
    """Uploads the contents of a local directory to S3.

    Args:
        local_directory (Path): The local directory to upload.
        bucket (str): The target S3 bucket name.
        s3_prefix (str): The prefix (folder path) within the S3 bucket.
    """
    # Check if directory exists and has content
    if not local_directory.is_dir() or not any(local_directory.iterdir()):
        logging.warning(f"Local directory {local_directory} does not exist or is empty. Skipping S3 upload.")
        return
        
    try:
        s3 = boto3.client('s3')
    except Exception as e:
        logging.error(f"Failed to create S3 client, skipping upload. Check credentials/config. Error: {e}")
        return
        
    logging.info(f"Attempting to upload contents of {local_directory} to s3://{bucket}/{s3_prefix}")
    num_uploaded = 0
    num_failed = 0

    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            local_path = Path(root) / filename
            # Create the relative path for the S3 key
            relative_path = local_path.relative_to(local_directory)
            s3_key = f"{s3_prefix.rstrip('/')}/{relative_path.as_posix()}"
            
            try:
                logging.debug(f"Uploading {local_path} to {s3_key}...")
                s3.upload_file(str(local_path), bucket, s3_key)
                num_uploaded += 1
            except ClientError as e:
                logging.error(f"Failed to upload {local_path} to S3: {e}")
                num_failed += 1
            except Exception as e:
                 logging.error(f"An unexpected error occurred during upload of {local_path}: {e}")
                 num_failed += 1
                 
    if num_failed == 0:
        logging.info(f"Successfully uploaded {num_uploaded} files to s3://{bucket}/{s3_prefix}")
    else:
        logging.warning(f"Completed upload attempt to s3://{bucket}/{s3_prefix} with {num_uploaded} successes and {num_failed} failures.")

def evaluate(model: AutoModelForCausalLM, dataloader: DataLoader, device: torch.device, current_epoch: int, args: argparse.Namespace, calculate_per_token_loss: bool = False):
    """Evaluates the model on the given dataloader.
    Returns average loss. If calculate_per_token_loss is True, also returns a list of per-token average losses.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # For per-token loss calculation
    # Initialize to full sequence_length as the plot x-axis is fixed
    per_token_losses_sum = torch.zeros(args.sequence_length, device=device) if calculate_per_token_loss else None
    per_token_counts = torch.zeros(args.sequence_length, device=device, dtype=torch.long) if calculate_per_token_loss else None
    
    with torch.no_grad():
        # current_epoch here is the 1-indexed evaluation epoch
        desc = f"Evaluation Epoch {current_epoch} Evaluating"
        if calculate_per_token_loss:
            desc += " (calc per-token loss)"
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch.to(device)
            labels = input_ids 
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss 
            total_loss += loss.item()
            num_batches += 1

            if calculate_per_token_loss and per_token_losses_sum is not None and per_token_counts is not None:
                # Logits for predicting the next token: outputs.logits is [batch_size, seq_len, vocab_size]
                # Labels are input_ids. Loss is calculated for predicting input_ids[i+1] from input_ids[0...i]
                # So, the loss for the token at labels[i] (0-indexed) is associated with logits output for position i-1.
                # The model internally shifts logits relative to labels.
                # outputs.logits are for *predicting* tokens input_ids[0]...input_ids[sequence_length-1]
                # Let input_ids be [t0, t1, ..., t_N-1] where N is sequence_length.
                # Logits for t0: outputs.logits[:, 0, :] (predicts t0, often from a BOS token or empty context, usually high loss or ignored)
                # Logits for t1: outputs.logits[:, 1, :] (predicts t1 based on t0)
                # ... 
                # Logits for t_N-1: outputs.logits[:, N-1, :] (predicts t_N-1 based on t0...t_N-2)
                
                # We want loss for each position in the sequence_length. Standard Causal LM loss is on predicting tokens 1 to N.
                # So, logits should be [batch, seq_len-1, vocab_size] and targets [batch, seq_len-1]
                
                logits_for_loss = outputs.logits[:, :-1, :].contiguous() # Input for predicting token 1 up to N (N-1 predictions in total)
                targets_for_loss = input_ids[:, 1:].contiguous()       # Actual tokens 1 up to N
                
                if logits_for_loss.size(1) == 0: # Should not happen if seq_len > 1
                    continue

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses_flat = loss_fct(logits_for_loss.view(-1, logits_for_loss.size(-1)), 
                                             targets_for_loss.view(-1))
                
                # Reshape to [batch_size, actual_prediction_len]
                # actual_prediction_len is sequence_length - 1
                actual_prediction_len = logits_for_loss.size(1)
                token_losses_per_batch_item = token_losses_flat.view(input_ids.size(0), actual_prediction_len)
                
                # token_losses_per_batch_item[b,j] is the loss for predicting the (j+1)-th token of the input sequence.
                # So j=0 is loss for predicting input_ids[b, 1], j=1 is loss for input_ids[b, 2] etc.
                # We want to store this in per_token_losses_sum where index i means loss for predicting token i of original sequence.
                # So, per_token_losses_sum[0] is loss for token 0 (often undefined/high)
                # per_token_losses_sum[1] is loss for token 1 (using token_losses_per_batch_item[:, 0])
                # per_token_losses_sum[i] is loss for token i (using token_losses_per_batch_item[:, i-1])

                # Sum per-position losses across the batch
                # Add to indices 1 to actual_prediction_len (inclusive) of per_token_losses_sum
                per_token_losses_sum[1 : actual_prediction_len + 1] += token_losses_per_batch_item.sum(dim=0)
                per_token_counts[1 : actual_prediction_len + 1] += input_ids.size(0) # Add batch_size for these positions

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    if not args.disable_wandb:
        # This logs the overall average validation loss for the evaluation epoch
        wandb.log({"eval/eval_epoch_val_loss": avg_loss, "eval_epoch": current_epoch}) # Changed from "epoch" to "eval_epoch"
    
    per_token_avg_losses_list = None
    if calculate_per_token_loss and per_token_losses_sum is not None and per_token_counts is not None:
        per_token_avg_losses_list = [float('nan')] * args.sequence_length # Initialize with NaNs
        # Iterate from token position 1 up to sequence_length -1 (as token 0 is not predicted from prior context here)
        for i in range(1, args.sequence_length):
            if per_token_counts[i] > 0:
                per_token_avg_losses_list[i] = (per_token_losses_sum[i] / per_token_counts[i]).item()
            # else it remains NaN, which is fine for plotting missing data
    return avg_loss, per_token_avg_losses_list

def train(args: argparse.Namespace):
    """Main training function."""
    # Create a unique run name
    run_name_base = f"k{args.k}_seed{args.seed}_lr{args.learning_rate}_bs{args.batch_size}"
    if args.run_suffix:
        run_name_base += f"_{args.run_suffix}"

    wandb_run_name = args.wandb_run_name if args.wandb_run_name else run_name_base

    output_dir_for_run = args.output_dir / run_name_base
    output_dir_for_run.mkdir(parents=True, exist_ok=True)
    step_checkpoints_dir = output_dir_for_run / "step_checkpoints"
    if args.checkpoint_interval_steps > 0:
        step_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Directory for top N EMA validation loss checkpoints
    top_ema_val_checkpoints_dir = output_dir_for_run / "top_ema_val_checkpoints"
    if args.num_best_ema_val_checkpoints > 1: # Only create if we are saving more than just the single 'best_model'
        top_ema_val_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    saved_top_ema_val_models_info = [] # Stores (ema_val_loss, path_str, optimizer_step, eval_epoch)

    # --- GEOM CKPT STATE --- (Now for raw validation loss based checkpoints)
    loss_ckpt_dir = output_dir_for_run / "loss_checkpoints"
    loss_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # New: Tracker for best raw validation loss for these specific checkpoints
    best_raw_val_for_geom_checkpoints = float('inf')
    # geom_ema, geom_last related to train loss EMA are no longer used for triggering these ckpts.
    geom_saved_checkpoints = [] # List of Path objects for rolling deletion of loss-based checkpoints
    # args.geom_alpha is likely unused. args.geom_beta (0.95 by default) is the improvement factor for raw validation loss.
    
    # For geometric checkpoint in-context loss plotting (still relevant when these ckpts are saved)
    geom_per_token_loss_xs = list(range(args.sequence_length)) 
    geom_per_token_loss_ys_list = []
    geom_per_token_loss_keys = []
    # ------------------------

    setup_logging(output_dir_for_run, run_name_base)
    save_args(args, output_dir_for_run, run_name_base)

    # Initialize W&B
    if not args.disable_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity, # Optional
                name=wandb_run_name,
                config=vars(args) # Log all hyperparameters
            )
            logging.info(f"Weights & Biases initialized for run: {wandb_run_name}, project: {args.wandb_project}")
        except Exception as e:
            logging.error(f"Failed to initialize Weights & Biases: {e}")
            logging.warning("Proceeding without W&B logging.")
            args.disable_wandb = True # Disable if init fails

    logging.info(f"Starting training run: {run_name_base}")
    logging.info(f"Output directory: {output_dir_for_run}")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) # For python's random ops (like category selection initial shuffle)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # torch.autograd.set_detect_anomaly(True) # Keep commented out unless actively debugging
    # logging.info("Enabled torch.autograd.set_detect_anomaly.") # Keep commented out

    # 1. Select K categories
    selected_train_categories = select_categories(ALL_CATEGORIES, k=args.k, seed=args.seed)
    logging.info(f"Selected {args.k} categories for training (seed {args.seed}): {selected_train_categories}")

    # Load Tokenizer early for vocab size check
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info(f"Tokenizer pad_token set to eos_token: {tokenizer.eos_token}")
        VOCAB_SIZE = tokenizer.vocab_size
    except Exception as e:
        logging.error(f"Failed to load tokenizer {args.model_name_or_path}: {e}")
        return

    # 2. Setup Datasets and DataLoaders
    logging.info("Setting up datasets and dataloaders...")
    preprocessed_data_path = args.preprocessed_data_dir

    try:
        train_dataset = RandomWindowDataset(
            preprocessed_dir=preprocessed_data_path,
            split="train",
            target_categories=selected_train_categories,
            sequence_length=args.sequence_length # Pass sequence length
        )
        val_dataset = RandomWindowDataset(
            preprocessed_dir=preprocessed_data_path,
            split="validation",
            target_categories=selected_train_categories, # Use same categories for validation
            sequence_length=args.sequence_length # Pass sequence length
        )
    except FileNotFoundError:
        logging.error(f"Preprocessed data not found in {preprocessed_data_path}. Please run fetch_arxiv.py first.")
        if not args.disable_wandb and wandb.run is not None: wandb.finish(exit_code=1)
        return
    except ValueError as e:
        logging.error(f"Error initializing dataset (maybe no papers for selected categories?): {e}")
        if not args.disable_wandb and wandb.run is not None: wandb.finish(exit_code=1)
        return

    if len(train_dataset) == 0:
        logging.error("Training dataset is empty after filtering. Exiting.")
        if not args.disable_wandb and wandb.run is not None: wandb.finish(exit_code=1)
        return
    if len(val_dataset) == 0:
        logging.warning("Validation dataset is empty after filtering. Proceeding without validation.")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, # Shuffle for training
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, # Can use a larger batch size for validation if memory allows
        shuffle=False, # No need to shuffle for validation
        num_workers=args.num_workers,
        pin_memory=True
    ) if len(val_dataset) > 0 else None

    logging.info(f"Train dataset size: {len(train_dataset)} (num batches per epoch: {len(train_dataloader)})")
    if val_dataloader:
        logging.info(f"Validation dataset size: {len(val_dataset)} (num batches per epoch: {len(val_dataloader)})")

    # 3. Load Model
    logging.info(f"Loading model: {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    original_model_config_dtype = getattr(config, 'torch_dtype', None)

    # For mixed precision (fp16, bf16) training with GradScaler (for fp16) or autocast alone (for bf16),
    # the model's master parameters should be in float32.
    # Autocast will then handle the execution of ops in the lower precision.
    # For true fp32 training, parameters are also float32.
    desired_param_dtype = torch.float32 

    if original_model_config_dtype != desired_param_dtype:
        logging.info(
            f"Model config for {args.model_name_or_path} has torch_dtype={original_model_config_dtype}. "
            f"Overriding to {desired_param_dtype} for parameter storage to ensure compatibility with mixed precision setup or for fp32 training."
        )
    config.torch_dtype = desired_param_dtype
    
    logging.info(f"Initializing model from config for {args.model_name_or_path} with parameters to be stored in {config.torch_dtype} (random initialization).")
    model = AutoModelForCausalLM.from_config(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}")
    model.to(device)

    # --- Initial/Manual Embedding Weight Init & Check ---
    try:
        if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'embed_in'):
            logging.info("Manually re-initializing embedding layer weights with smaller stddev (0.01)")
            std_dev = 0.01 # Smaller std dev for potentially more stable init
            model.gpt_neox.embed_in.weight.data.normal_(mean=0.0, std=std_dev)
            logging.info(f"Embedding weights re-initialized using normal_(mean=0.0, std={std_dev})")
            
            embed_weights_to_check = model.gpt_neox.embed_in.weight # Corrected variable name
            logging.info(f"INITIAL CHECK: Checking embedding weights ({embed_weights_to_check.shape}) after manual re-init")
            is_init_weights_finite = torch.isfinite(embed_weights_to_check).all().item()
            logging.info(f"INITIAL CHECK: Embedding weights finite: {is_init_weights_finite}")
            if not is_init_weights_finite:
                logging.error("INITIAL CHECK FAILED: Embedding weights NON-FINITE after manual re-init!")
            else:
                logging.info(f"INITIAL CHECK: Weights Min: {embed_weights_to_check.min().item():.4f}")
                logging.info(f"INITIAL CHECK: Weights Max: {embed_weights_to_check.max().item():.4f}")
                # Log mean as well
                logging.info(f"INITIAL CHECK: Weights Mean: {embed_weights_to_check.mean().item():.4f}")
        else:
             logging.warning("INITIAL CHECK: Could not find embedding layer for re-init/check.")
    except Exception as e_init_check:
        logging.error(f"INITIAL CHECK: Error during embedding weight re-init/check: {e_init_check}")
    # --- END Init & Check ---

    # Log initial model parameter norm
    initial_model_param_norm = 0
    for p in model.parameters():
        initial_model_param_norm += p.data.norm(2).item() ** 2
    initial_model_param_norm = initial_model_param_norm ** 0.5
    logging.info(f"Initial Model Parameter Norm: {initial_model_param_norm:.4f}")
    if not args.disable_wandb:
        wandb.summary["initial_model_param_norm"] = initial_model_param_norm

    # 4. Optimizer and Scheduler
    logging.info("Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=args.learning_rate, 
                                weight_decay=args.weight_decay, 
                                eps=args.adam_epsilon,
                                betas=(args.adam_beta1, args.adam_beta2)) # Use configured betas
    
    # Calculate total training steps based on evaluation epochs and steps per eval epoch
    max_total_optimizer_steps = args.epochs * args.steps_per_eval_epoch
    logging.info(f"Max total optimizer steps: {max_total_optimizer_steps} ({args.epochs} eval epochs * {args.steps_per_eval_epoch} steps/eval_epoch)")

    if args.num_warmup_steps > max_total_optimizer_steps:
        logging.warning(f"num_warmup_steps ({args.num_warmup_steps}) is greater than max_total_optimizer_steps ({max_total_optimizer_steps}). Consider reducing warmup steps.")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
            optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps, # Scheduler steps are micro-steps if it depends on total steps
        num_training_steps=max_total_optimizer_steps * args.gradient_accumulation_steps # Scheduler steps are micro-steps
    )
    logging.info(f"LR Scheduler: {args.lr_scheduler_type} with {args.num_warmup_steps * args.gradient_accumulation_steps} micro warmup steps and {max_total_optimizer_steps * args.gradient_accumulation_steps} total micro training steps.")

    # Gradient scaler for mixed precision (fp16)
    scaler = GradScaler(enabled=(args.precision == "fp16"))
    logging.info(f"GradScaler enabled: {scaler.is_enabled()}")

    # For "best model" saving based on EMA validation loss
    best_ema_val_loss = float('inf')
    epochs_since_last_best_ema_val_loss = 0 # For early stopping based on eval epochs
    # Note: The single best_model/ checkpoint is handled separately from the top_N list.
    
    # For ReduceLROnPlateau
    plateau_scheduler = None
    if args.reduce_lr_factor < 1.0:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.min_lr,
            verbose=True # Enable verbose output if scheduler is active
        )
        logging.info(f"ReduceLROnPlateau scheduler ACTIVATED with factor={args.reduce_lr_factor}, patience={args.reduce_lr_patience}, min_lr={args.min_lr}.")
    else:
        logging.info(f"ReduceLROnPlateau scheduler DISABLED (factor={args.reduce_lr_factor} >= 1.0).")


    # 5. Training Loop
    logging.info("Starting training loop...")
    global_optimizer_step = 0
    global_micro_step = 0 # Tracks total micro-steps (samples processed * sequence_length)
    
    # Use an iterator for the dataloader to allow cycling
    train_dataloader_iter = iter(train_dataloader)
    
    # Progress bar for total optimizer steps
    progress_bar_total_steps = tqdm(total=max_total_optimizer_steps, desc="Total Optimizer Steps", unit="step", dynamic_ncols=True)

    # Training loop (while optimizer steps < max_total_optimizer_steps)
    # Loop condition is at the end of the inner loop to ensure final evaluation and saving
    # current_eval_epoch is 1-indexed for logging/display
    current_eval_epoch = 0 

    training_start_time = time.time()

    try:
        while global_optimizer_step < max_total_optimizer_steps:
            model.train() # Set model to training mode for each "conceptual" epoch start
            
            # Inner loop for optimizer steps within an evaluation period
            # This loop will run args.steps_per_eval_epoch times OR until max_total_optimizer_steps is reached
            steps_in_current_eval_period = 0
            while steps_in_current_eval_period < args.steps_per_eval_epoch and global_optimizer_step < max_total_optimizer_steps:
                optimizer.zero_grad() # Zero gradients for each accumulation cycle
                
                accumulated_loss = 0.0 # Accumulate loss over gradient_accumulation_steps
                
                for micro_step in range(args.gradient_accumulation_steps):
                    try:
                        batch = next(train_dataloader_iter)
                    except StopIteration:
                        logging.info("Training dataloader exhausted, re-initializing.")
                        train_dataloader_iter = iter(train_dataloader)
                        batch = next(train_dataloader_iter)
                    
                    # Assuming batch is the tensor itself due to IndexError
                    # input_ids = batch['input_ids'].to(device) 
                    input_ids = batch.to(device)
                    labels = input_ids.clone() # For causal LM, labels are usually input_ids shifted

                    # Forward pass with autocast if precision is not fp32
                    with autocast(enabled=(args.precision != "fp32"), dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16):
                        outputs = model(input_ids, labels=labels)
                        loss = outputs.loss
                        
                        # Logit check (example, can be expanded)
                        if not torch.isfinite(outputs.logits).all():
                            logging.error(f"NON-FINITE LOGITS DETECTED at global optimizer step {global_optimizer_step}, micro_step in acc: {micro_step}")
                            # Potentially save debug info, then raise error or try to recover
                            # For now, log and continue, but this often precedes NaN loss
                            if not args.disable_wandb and wandb.run is not None:
                                wandb.log({"error/non_finite_logits_step": global_optimizer_step})


                    if loss is None or not torch.isfinite(loss):
                        logging.error(f"NaN or Inf loss detected BEFORE backward() at global_optimizer_step={global_optimizer_step}, micro_step_in_acc={micro_step}. Loss: {loss}")
                        if not args.disable_wandb and wandb.run is not None:
                            wandb.log({"error/nan_inf_loss_step": global_optimizer_step})
                        # Consider skipping optimizer step or stopping if NaNs are persistent
                        # For now, we will let it proceed to scaler.scale(loss).backward() which might also error
                        # and then the optimizer step might be skipped if grads are non-finite.
                        # If loss is None, backward() will fail.
                        if loss is None: loss = torch.tensor(float('nan'), device=device) # Ensure loss is a tensor for backward() to fail on if it was None
                    
                    # Normalize loss if accumulating gradients
                    loss = loss / args.gradient_accumulation_steps
                    accumulated_loss += loss.item() # Accumulate item for logging
                    
                    # Backward pass
                    scaler.scale(loss).backward() # Scales loss and calls backward
                    
                    global_micro_step += 1

                # Gradient Clipping (on unscaled grads)
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer Step
                scaler.step(optimizer) # Optimizer step, checks for inf/NaN scaled grads
                scaler.update() # Update scaler for next iteration
                
                lr_scheduler.step() # Step LR scheduler

                global_optimizer_step += 1
                steps_in_current_eval_period += 1
                progress_bar_total_steps.update(1)

                # Logging at log_interval (optimizer steps)
                if global_optimizer_step % args.log_interval == 0:
                    log_payload = {
                        "train/loss": accumulated_loss, # Avg loss for this optimizer step
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "progress/optimizer_step": global_optimizer_step,
                        "progress/micro_step": global_micro_step,
                        "progress/current_eval_epoch": current_eval_epoch + 1 # 1-indexed for display
                    }
                    logging.info(
                        f"Eval Epoch: {current_eval_epoch + 1}/{args.epochs} | "
                        f"Opt Step: {global_optimizer_step}/{max_total_optimizer_steps} | "
                        f"Train Loss: {accumulated_loss:.4f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )
                    if not args.disable_wandb and wandb.run is not None:
                        wandb.log(log_payload, step=global_optimizer_step)
                
                # Periodic step-based checkpointing (based on optimizer steps)
                if args.checkpoint_interval_steps > 0 and global_optimizer_step % args.checkpoint_interval_steps == 0:
                    ckpt_path = step_checkpoints_dir / f"step_{global_optimizer_step:06d}"
                    model.save_pretrained(ckpt_path)
                    logging.info(f"Saved periodic step-based checkpoint to {ckpt_path} at optimizer step {global_optimizer_step}")
                    # Manage number of step checkpoints
                    existing_step_ckpts = sorted(step_checkpoints_dir.glob("step_*"), key=os.path.getmtime)
                    if len(existing_step_ckpts) > args.max_step_checkpoints:
                        for old_ckpt in existing_step_ckpts[:-args.max_step_checkpoints]:
                            logging.info(f"Removing old step checkpoint: {old_ckpt}")
                            shutil.rmtree(old_ckpt)
                
                if global_optimizer_step >= max_total_optimizer_steps:
                    break # Exit inner loop if max total steps reached

            # End of an evaluation period (args.steps_per_eval_epoch completed or max_total_optimizer_steps reached)
            current_eval_epoch += 1 # Increment evaluation epoch counter (1-indexed for external use)
            logging.info(f"Completed evaluation period. Current evaluation epoch: {current_eval_epoch}")

            ema_val_loss_for_plateau_scheduler = best_ema_val_loss # Default to best if no new eval

            if val_dataloader:
                logging.info(f"Evaluating on validation set for evaluation epoch {current_eval_epoch} (OptStep: {global_optimizer_step})...")
                # Pass current_eval_epoch (1-indexed) to evaluate function
                raw_val_loss, _ = evaluate(model, val_dataloader, device, current_eval_epoch, args, calculate_per_token_loss=False) 
                
                if raw_val_loss is None or not np.isfinite(raw_val_loss): # Check for finite raw_val_loss
                    logging.warning(f"Validation loss is NaN or Inf at eval epoch {current_eval_epoch}. Skipping best model/early stopping/loss-checkpoint logic for this epoch.")
                else:
                    # Update EMA of validation loss (for best_model checkpointing and ReduceLROnPlateau)
                    if best_ema_val_loss == float('inf'): # First validation
                        current_ema_val_loss = raw_val_loss
                    else:
                        val_ema_alpha = 0.2 # Smoothing factor for EMA validation loss (can be an arg)
                        current_ema_val_loss = val_ema_alpha * raw_val_loss + (1 - val_ema_alpha) * best_ema_val_loss
                    
                    logging.info(f"Eval Epoch: {current_eval_epoch}/{args.epochs} | Opt Step: {global_optimizer_step} | Raw Val Loss: {raw_val_loss:.4f} | EMA Val Loss: {current_ema_val_loss:.4f}")
                    if not args.disable_wandb and wandb.run is not None:
                        wandb.log({
                            "eval/raw_validation_loss": raw_val_loss,
                            "eval/ema_validation_loss": current_ema_val_loss,
                            "eval/evaluation_epoch": current_eval_epoch
                        }, step=global_optimizer_step)

                    ema_val_loss_for_plateau_scheduler = current_ema_val_loss # Used by ReduceLROnPlateau

                    # Check for "best model" based on EMA validation loss (always keeps the single best overall)
                    if current_ema_val_loss < best_ema_val_loss:
                        best_ema_val_loss = current_ema_val_loss
                        epochs_since_last_best_ema_val_loss = 0
                        best_model_path = output_dir_for_run / "best_model"
                        model.save_pretrained(best_model_path)
                        logging.info(f"New best overall EMA validation loss: {best_ema_val_loss:.4f}. Saved best_model to {best_model_path}")
                        if not args.disable_wandb and wandb.run is not None:
                            wandb.summary["best_ema_val_loss"] = best_ema_val_loss
                            wandb.summary["best_ema_val_loss_optimizer_step"] = global_optimizer_step
                            wandb.summary["best_ema_val_loss_eval_epoch"] = current_eval_epoch
                    else:
                        epochs_since_last_best_ema_val_loss += 1
                        logging.info(f"EMA validation loss did not improve for {epochs_since_last_best_ema_val_loss} eval epochs. Best overall EMA: {best_ema_val_loss:.4f}")

                    # --- Top N EMA Validation Loss Checkpoints --- 
                    if args.num_best_ema_val_checkpoints > 1: # Only active if user wants more than the single best_model
                        made_change_to_top_list = False
                        # Check if current model is better than the worst in the list or if list is not full
                        if len(saved_top_ema_val_models_info) < args.num_best_ema_val_checkpoints or \
                           current_ema_val_loss < saved_top_ema_val_models_info[-1][0]: # Assumes list is sorted best to worst
                            
                            # Create path for this potential top N checkpoint
                            # Use a descriptive name including loss, step, and epoch
                            top_n_ckpt_name = f"ema_{current_ema_val_loss:.4f}_step_{global_optimizer_step}_epoch_{current_eval_epoch}"
                            top_n_ckpt_path = top_ema_val_checkpoints_dir / top_n_ckpt_name
                            
                            model.save_pretrained(top_n_ckpt_path)
                            logging.info(f"Saved top N EMA val checkpoint to {top_n_ckpt_path} (EMA: {current_ema_val_loss:.4f})")
                            
                            # Add to list
                            saved_top_ema_val_models_info.append((current_ema_val_loss, str(top_n_ckpt_path), global_optimizer_step, current_eval_epoch))
                            # Sort by EMA val loss (ascending - best first)
                            saved_top_ema_val_models_info.sort(key=lambda x: x[0])
                            made_change_to_top_list = True

                            # If list is now too long, remove the worst one
                            if len(saved_top_ema_val_models_info) > args.num_best_ema_val_checkpoints:
                                worst_model_info = saved_top_ema_val_models_info.pop() # Removes the last (worst)
                                worst_model_path_to_delete = Path(worst_model_info[1])
                                if worst_model_path_to_delete.exists():
                                    shutil.rmtree(worst_model_path_to_delete)
                                    logging.info(f"Removed old top N EMA val checkpoint: {worst_model_path_to_delete} (EMA: {worst_model_info[0]:.4f})")
                        
                        if made_change_to_top_list and not args.disable_wandb and wandb.run is not None:
                            # Log the current top N list (e.g., just the losses and steps)
                            top_n_summary = [(info[0], info[2], info[3]) for info in saved_top_ema_val_models_info] # (loss, step, epoch)
                            wandb.log({"eval/top_n_ema_val_checkpoints_summary": top_n_summary}, step=global_optimizer_step)

                    # --- "Loss-Based" Checkpoint (now triggered by raw validation loss improvement) ---
                    # args.max_loss_ckpts: 0 means unlimited, negative means disabled.
                    if args.max_loss_ckpts >= 0:
                        # Check if raw_val_loss is significantly better than the best raw val loss seen by *this* mechanism
                        if raw_val_loss < (best_raw_val_for_geom_checkpoints * args.geom_beta):
                            logging.info(f"Raw validation loss {raw_val_loss:.4f} meets criteria for loss-based checkpoint against best_raw_val_for_geom ({best_raw_val_for_geom_checkpoints:.4f}) with factor {args.geom_beta}.")
                            
                            # save_geom_ckpt expects `loss_val` (the metric value causing the save), 
                            # `current_epoch_idx` (0-indexed for its internal display/use if it expects an index),
                            # and other context.
                            save_geom_ckpt(
                                model=model, 
                                step=global_optimizer_step, 
                                loss_val=raw_val_loss, # This is the raw validation loss that triggered the save
                                loss_ckpt_dir=loss_ckpt_dir, 
                                geom_saved_list=geom_saved_checkpoints, # Pass the correct list
                                args_ref=args, 
                                wandb_ref=wandb, 
                                logging_ref=logging, 
                                shutil_ref=shutil,
                                val_dataloader=val_dataloader, 
                                device_ref=device, 
                                current_epoch_idx=current_eval_epoch - 1, # Pass 0-indexed if current_eval_epoch is 1-indexed
                                evaluate_fn=evaluate, 
                                geom_xs=geom_per_token_loss_xs, 
                                geom_ys_list=geom_per_token_loss_ys_list, 
                                geom_keys=geom_per_token_loss_keys,
                                output_dir_for_run_ref=output_dir_for_run
                            )
                            best_raw_val_for_geom_checkpoints = raw_val_loss # Update the best raw val for this mechanism
                        else:
                            logging.info(f"Raw validation loss {raw_val_loss:.4f} did not meet criteria for loss-based checkpoint against best_raw_val_for_geom ({best_raw_val_for_geom_checkpoints:.4f}) with factor {args.geom_beta}.")
                    # --- End of "Loss-Based" Checkpoint logic ---


            # Step the ReduceLROnPlateau scheduler with EMA validation loss
            if plateau_scheduler and val_dataloader and np.isfinite(ema_val_loss_for_plateau_scheduler): # ensure it's a finite number
                plateau_scheduler.step(ema_val_loss_for_plateau_scheduler)
            elif not val_dataloader:
                logging.debug("No validation dataloader, skipping ReduceLROnPlateau scheduler step.")
            elif not plateau_scheduler:
                logging.debug("ReduceLROnPlateau scheduler is disabled, skipping step.")


            # Early stopping check (based on EMA validation loss and eval epochs)
            if args.early_stopping_patience > 0 and epochs_since_last_best_ema_val_loss >= args.early_stopping_patience:
                logging.info(f"Early stopping triggered after {args.early_stopping_patience} evaluation epochs with no improvement in EMA validation loss.")
                if not args.disable_wandb and wandb.run is not None:
                    wandb.summary["stopped_early"] = True
                    wandb.summary["stopped_at_eval_epoch"] = current_eval_epoch 
                    wandb.summary["stopped_at_optimizer_step"] = global_optimizer_step
                break # Exit the main training loop (while global_optimizer_step < max_total_optimizer_steps)
            
            if global_optimizer_step >= max_total_optimizer_steps:
                logging.info(f"Reached max_total_optimizer_steps ({max_total_optimizer_steps}).")
                break

        # End of training loop (while global_optimizer_step < max_total_optimizer_steps)
        progress_bar_total_steps.close()
        training_duration_seconds = time.time() - training_start_time
        logging.info(f"Training finished. Total optimizer steps: {global_optimizer_step}. Total evaluation epochs: {current_eval_epoch}.")
        logging.info(f"Total training time: {training_duration_seconds:.2f} seconds ({training_duration_seconds/3600:.2f} hours).")

        if not args.disable_wandb and wandb.run is not None:
            wandb.summary["total_optimizer_steps_completed"] = global_optimizer_step
            wandb.summary["total_eval_epochs_completed"] = current_eval_epoch
            wandb.summary["training_duration_hours"] = training_duration_seconds / 3600
            if "stopped_early" not in wandb.summary: # If not stopped early
                 wandb.summary["stopped_early"] = False


    except Exception as e:
        logging.error(f"Exception during training: {e}", exc_info=True) # Log traceback
        if not args.disable_wandb and wandb.run is not None:
            wandb.log({"error/training_exception": str(e)}, step=global_optimizer_step)
            wandb.summary["training_crashed"] = True
            wandb.summary["crash_message"] = str(e)
    finally:
        # Save final model
        final_model_path = output_dir_for_run / "final_model"
        
        should_save_best_model_as_final = False
        if not args.disable_wandb and wandb.run is not None:
            # Safely check wandb.summary for early stopping information
            try:
                if wandb.summary.get("stopped_early", False): # Use .get for safety
                    should_save_best_model_as_final = True
            except Exception as e_summary_check:
                logging.warning(f"Could not access wandb.summary for early stopping check: {e_summary_check}. Proceeding to save current model.")

        if should_save_best_model_as_final:
            # If stopped early (and W&B summary was accessible), the best_model is the one to save as final
            best_model_path = output_dir_for_run / "best_model"
            if best_model_path.exists():
                logging.info(f"Early stopping indicated: Using best model from {best_model_path} as the final model.")
                try:
                    logging.info(f"Reloading best model from {best_model_path} to save as final model.")
                    reloaded_best_model_config = AutoConfig.from_pretrained(best_model_path)
                    reloaded_best_model = AutoModelForCausalLM.from_pretrained(best_model_path, config=reloaded_best_model_config)
                    reloaded_best_model.save_pretrained(final_model_path)
                    logging.info(f"Saved reloaded best model to {final_model_path}")
                except Exception as e_reload_save:
                    logging.error(f"Could not reload and save best_model as final_model. Saving current model instead. Error: {e_reload_save}")
                    model.save_pretrained(final_model_path) # Fallback to current model
                    logging.info(f"Saved current model as final model to {final_model_path} (fallback).")
            else: # This else corresponds to 'if best_model_path.exists():'
                logging.warning(f"Early stopping indicated, but no best_model found at {best_model_path}. Saving current model as final.")
                model.save_pretrained(final_model_path)
                logging.info(f"Saved current model as final model to {final_model_path}")
        else:
            # Training completed normally, crashed before early stopping determination, W&B disabled, or W&B summary access failed.
            # Save current model state.
            logging.info(f"Not using best_model from early stopping (or info unavailable). Saving current model state as final_model to {final_model_path}")
            model.save_pretrained(final_model_path)
            logging.info(f"Saved current model as final model to {final_model_path}")


        # Upload results to S3 if configured
        if args.upload_results_to_s3 and args.s3_bucket and args.s3_prefix:
            logging.info(f"Uploading results from {output_dir_for_run} to S3 bucket {args.s3_bucket} with prefix {args.s3_prefix}...")
            upload_directory_to_s3(output_dir_for_run, args.s3_bucket, args.s3_prefix)
            
            if args.delete_local_checkpoints_after_s3_upload:
                logging.info("Deleting local checkpoint directories after S3 upload...")
                if step_checkpoints_dir.exists():
                    shutil.rmtree(step_checkpoints_dir)
                    logging.info(f"Deleted local step checkpoints: {step_checkpoints_dir}")
                if loss_ckpt_dir.exists(): # This is the one for "geom" / raw_val checkpoints
                    shutil.rmtree(loss_ckpt_dir)
                    logging.info(f"Deleted local loss checkpoints: {loss_ckpt_dir}")
                if args.num_best_ema_val_checkpoints > 1 and top_ema_val_checkpoints_dir.exists():
                    shutil.rmtree(top_ema_val_checkpoints_dir)
                    logging.info(f"Deleted local top EMA validation checkpoints: {top_ema_val_checkpoints_dir}")
                # Keep best_model and final_model locally for now, or add specific flags for them
                # best_model_path = output_dir_for_run / "best_model"
                # if best_model_path.exists():
                #     shutil.rmtree(best_model_path)
                #     logging.info(f"Deleted local best model: {best_model_path}")

        if not args.disable_wandb and wandb.run is not None:
            wandb.finish()
            logging.info("Weights & Biases run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Causal Language Model on ArXiv data with category selection.")

    # Path arguments
    parser.add_argument("--preprocessed_data_dir", type=Path, default=DEFAULT_PREPROCESSED_DIR, help="Directory containing preprocessed data (tokens.bin, index.jsonl, splits.json).")
    parser.add_argument("--output_dir", type=Path, default=Path("./training_output"), help="Root directory to save training outputs (models, logs).")
    parser.add_argument("--run_suffix", type=str, default=None, help="Optional suffix to append to the run name and output directory.")

    # Model and Tokenizer arguments
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model name or path to local model/config.")
    
    # Dataset and Dataloader arguments
    parser.add_argument("--k", type=int, default=3, help="Number of categories to select for training.")
    parser.add_argument("--sequence_length", type=int, default=256, help="Sequence length for model input.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader. Set to 0 if experiencing issues with too many open files, especially on macOS.") # Updated default and help

    # Training configuration arguments
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"], help="Training precision.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device (micro-batch size). Effective batch size is batch_size * gradient_accumulation_steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before an optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type (e.g., 'linear', 'cosine', 'constant').")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of warmup steps for the learning rate scheduler (applied to micro-steps).")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1 parameter.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2 parameter.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon parameter.") # User confirmed 1e-8 is okay for final runs
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping.")
    
    # Epochs and Steps Arguments (Refactored for Evaluation Epochs)
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of evaluation epochs to train for.") # Renamed from max_epochs_hard_limit
    parser.add_argument("--steps_per_eval_epoch", type=int, default=100, help="Number of optimizer steps that constitute one 'evaluation epoch'. Evaluation and related actions (early stopping, LR reduction) happen after this many steps.")

    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=100, help="Log training metrics every N optimizer steps.")

    # Checkpoint arguments
    parser.add_argument("--checkpoint_interval_steps", type=int, default=1000, help="Save a checkpoint every N optimizer steps. 0 or negative to disable.")
    parser.add_argument("--max_step_checkpoints", type=int, default=5, help="Maximum number of periodic step-based checkpoints to keep.")
    
    # Best EMA validation loss model checkpoints
    parser.add_argument("--num_best_ema_val_checkpoints", type=int, default=1, help="Number of top models based on EMA validation loss to save. The single best is always saved to 'best_model/'. This saves additional ones to 'top_ema_val_checkpoints/'. Default 1 means no extra copies beyond 'best_model/'. Set to >1 to keep multiple (e.g., 5).")

    # Geometric loss checkpoint arguments (now based on raw validation loss)
    parser.add_argument("--max_loss_ckpts", type=int, default=0, help="Maximum number of loss-based checkpoints to keep (0 for unlimited, negative to disable). These are triggered by raw validation loss improvements.") # Clarified help for max_loss_ckpts
    parser.add_argument("--geom_beta", type=float, default=0.95, help="Factor for raw validation loss improvement to trigger a loss-based checkpoint. Saves if current_raw_val_loss < best_raw_val_for_geom_checkpoints * geom_beta (e.g., 0.95 means 5% improvement needed).") # Updated help

    # Early stopping arguments
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Number of evaluation epochs with no improvement in EMA validation loss after which training will be stopped.")
    parser.add_argument("--early_stopping_delta", type=float, default=0.0, help="Minimum change in EMA validation loss to qualify as an improvement for early stopping.")
    
    # ReduceLROnPlateau arguments
    parser.add_argument("--reduce_lr_factor", type=float, default=1.0, help="Factor by which the learning rate will be reduced by ReduceLROnPlateau (1.0 effectively disables it).") # Default 1.0
    parser.add_argument("--reduce_lr_patience", type=int, default=10, help="Number of evaluation epochs with no EMA validation loss improvement after which learning rate will be reduced (if factor < 1.0).")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum learning rate for ReduceLROnPlateau.")


    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--force_cpu", action="store_true", help="Force training on CPU even if CUDA is available.")

    # W&B arguments
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="icl-non-ergodic-arxiv", help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (username or team). Optional.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom run name for Weights & Biases. If None, a name is generated.")

    # S3 Upload arguments
    parser.add_argument("--upload_results_to_s3", action="store_true", help="Upload final results and checkpoints to S3.")
    parser.add_argument("--s3_bucket", type=str, default=os.getenv("S3_RESULTS_BUCKET"), help="S3 bucket for uploading results.")
    parser.add_argument("--s3_prefix", type=str, default=os.getenv("S3_RESULTS_PREFIX"), help="S3 prefix (folder path) for uploading results within the bucket.")
    parser.add_argument("--delete_local_checkpoints_after_s3_upload", action="store_true", help="Delete local step_checkpoints and loss_checkpoints directories after successful S3 upload to save space.")

    
    args = parser.parse_args()

    # Further refine help text for clarity based on final decision for max_loss_ckpts
    # If args.max_loss_ckpts is 0, it means unlimited. If negative, it's disabled.
    # The current help text says "0 for unlimited, negative to disable".
    # The code for triggering uses `if args.max_loss_ckpts >= 0:`. This means 0 IS unlimited, and negative values disable it. This is consistent.

    # Update Adam Epsilon if needed based on notes.md or specific run script settings
    # From notes.md, the NaN was due to 1e-8 with Pythia on first step, changed to 1e-6 for stability.
    # However, scripts/run_multi_gpu_final.sh has ADAM_EPSILON=1e-8
    # For final runs, user confirmed 1e-8 is okay. So default 1e-8 in parser is fine.

    train(args) 