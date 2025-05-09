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
        for batch in tqdm(dataloader, desc=f"Epoch {current_epoch+1} Evaluating{' (calc per-token loss)' if calculate_per_token_loss else ''}", leave=False):
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
        # This logs the overall average validation loss for the epoch
        wandb.log({"eval/epoch_val_loss": avg_loss, "epoch": current_epoch + 1})
    
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

    # --- GEOM CKPT STATE ---
    loss_ckpt_dir = output_dir_for_run / "loss_checkpoints"
    loss_ckpt_dir.mkdir(parents=True, exist_ok=True)

    geom_ema   = None        # running EMA of train loss
    geom_last  = None        # EMA at last checkpoint
    geom_saved = []          # list of Path objects for rolling deletion
    alpha, beta = args.geom_alpha, args.geom_beta
    # For geometric checkpoint in-context loss plotting
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

    # Keep detect_anomaly enabled
    torch.autograd.set_detect_anomaly(True)
    logging.info("Enabled torch.autograd.detect_anomaly.")

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
    # Use requested beta2 = 0.95 and eps = 1e-8
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=args.learning_rate, 
                                weight_decay=args.weight_decay, 
                                eps=args.adam_epsilon, # Use arg
                                betas=(args.adam_beta1, args.adam_beta2)) # Use args
    
    # Determine total training steps (optimizer steps)
    if args.token_budget > 0:
        # Note: sequence_length replaces EFFECTIVE_WINDOW_SIZE
        tokens_per_optimizer_step = args.batch_size * args.sequence_length * args.gradient_accumulation_steps
        if tokens_per_optimizer_step == 0: 
            logging.error("Cannot calculate steps: tokens_per_optimizer_step is zero.")
            if not args.disable_wandb and wandb.run is not None: wandb.finish(exit_code=1)
            return
        # This is the total number of OPTIMIZER steps
        num_total_training_steps = args.token_budget // tokens_per_optimizer_step 
        logging.info(f"Token budget: {args.token_budget} tokens. Gradient Acc Steps: {args.gradient_accumulation_steps}. Calculated total optimizer steps: {num_total_training_steps}")
    else:
        # If epoch-based, adjust total steps for gradient accumulation
        num_micro_batches_per_epoch = len(train_dataloader)
        num_optimizer_steps_per_epoch = num_micro_batches_per_epoch // args.gradient_accumulation_steps
        num_total_training_steps = args.epochs * num_optimizer_steps_per_epoch
        logging.info(f"Epoch-based training. Total epochs: {args.epochs}. Grad Acc Steps: {args.gradient_accumulation_steps}. Max optimizer steps: {num_total_training_steps}")
    
    if num_total_training_steps == 0 and (args.epochs > 0 or args.token_budget > 0) and len(train_dataloader) > 0:
        logging.warning("Calculated num_total_training_steps is 0. This can happen if token_budget is too small for even one step, or if epochs=0 and token_budget=0. Training will not run effectively.")
        # Allow to proceed if epochs and token_budget are both 0 (e.g. for a quick model load test), but log it.
        if args.epochs == 0 and args.token_budget == 0:
            logging.info("Epochs and Token Budget are both 0. No training steps will be performed.")
        # else: # If one of them was >0 but still resulted in 0 steps, it's more of an issue.
            # if not args.disable_wandb: wandb.finish(exit_code=1)
            # return # Optionally exit

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_total_training_steps # Scheduler uses optimizer steps
    )

    # 5. Training Loop
    logging.info("Starting training loop...")
    global_optimizer_step = 0
    global_micro_batch_step = 0 # Counts micro-batches *processed* (i.e., after backward pass) within an accumulation cycle
    best_val_loss = float('inf')
    saved_step_checkpoints = []
    training_complete = False
    first_batch_checked = False # Flag for vocab check
    accumulated_loss_for_opt_step = 0.0 # Accumulator for average loss
    last_opt_step_time = time.time() # For timing steps
    scaler = torch.amp.GradScaler(enabled=(args.precision == 'fp16')) # GradScaler for fp16
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32 # Determine autocast dtype
    final_saved_model_path = output_dir_for_run / "final_model" # Define path earlier

    # --- Embedding Debug Hook ---
    nan_debug_info = {'triggered': False, 'hook_active': True} # Control hook logging
    
    # Define hook function within train() to close over relevant variables like VOCAB_SIZE, 
    # global_optimizer_step, global_micro_batch_step, and nan_debug_info.
    # Note: VOCAB_SIZE must be defined in the outer scope (train function) before this.
    def embedding_forward_hook(module, input_args, output):
        if not nan_debug_info['hook_active']: # Allow disabling the hook's logic
            return output

        # Check if this is the target micro-batch of the 1st optimizer step.
        # global_micro_batch_step is 0-indexed for micro-batches within an accumulation cycle.
        # It reflects the state *before* the current micro-batch's forward/backward.
        # So, for the 8th micro-batch (if accum_steps=8), global_micro_batch_step will be 7.
        is_target_step = (global_optimizer_step == 0 and 
                          global_micro_batch_step == (args.gradient_accumulation_steps - 1) )

        if is_target_step and not nan_debug_info['triggered']:
            try:
                input_ids_to_embedding = input_args[0]
                output_embeddings = output

                logging.info(f"HOOK (Opt_Step={global_optimizer_step}, Micro_Step_About_To_Process={global_micro_batch_step+1}): Embedding Layer Forward Hook Triggered.")
                logging.info(f"HOOK: Input IDs to embedding layer shape: {input_ids_to_embedding.shape}, dtype: {input_ids_to_embedding.dtype}, device: {input_ids_to_embedding.device}")
                input_ids_min = input_ids_to_embedding.min().item()
                input_ids_max = input_ids_to_embedding.max().item()
                logging.info(f"HOOK: Input IDs min: {input_ids_min}, max: {input_ids_max}")

                if torch.any(input_ids_to_embedding < 0) or torch.any(input_ids_to_embedding >= VOCAB_SIZE):
                    logging.error(f"HOOK: Problematic input IDs DETECTED: min={input_ids_min}, max={input_ids_max}, vocab_size={VOCAB_SIZE}")
                    problematic_ids_mask = (input_ids_to_embedding < 0) | (input_ids_to_embedding >= VOCAB_SIZE)
                    problematic_ids_values = input_ids_to_embedding[problematic_ids_mask]
                    logging.error(f"HOOK: Specific out-of-bounds/negative IDs ({problematic_ids_values.numel()}): {problematic_ids_values.tolist()[:20]}") # Log first 20

                logging.info(f"HOOK: Embedding layer weights shape: {module.weight.shape}, dtype: {module.weight.dtype}, device: {module.weight.device}")
                logging.info(f"HOOK: Embedding layer weights (sample, weight[0, :10]): {module.weight.data[0, :10].tolist()}")
                weights_are_finite = torch.isfinite(module.weight.data).all().item()
                logging.info(f"HOOK: Embedding layer weights finite: {weights_are_finite}")
                if weights_are_finite:
                    logging.info(f"HOOK: Embedding layer weights min: {module.weight.data.min().item():.4e}, max: {module.weight.data.max().item():.4e}, mean: {module.weight.data.mean().item():.4e}")
                else:
                    logging.error("HOOK: EMBEDDING WEIGHTS ARE NON-FINITE!")
                    num_non_finite_weights = (~torch.isfinite(module.weight.data)).sum().item()
                    logging.error(f"HOOK: Number of non-finite weights: {num_non_finite_weights} / {module.weight.data.numel()}")


                logging.info(f"HOOK: Output embeddings shape: {output_embeddings.shape}, dtype: {output_embeddings.dtype}, device: {output_embeddings.device}")
                outputs_are_finite = torch.isfinite(output_embeddings).all().item()
                logging.info(f"HOOK: Output embeddings finite: {outputs_are_finite}")
                if outputs_are_finite:
                    logging.info(f"HOOK: Output embeddings min: {output_embeddings.min().item():.4e}, max: {output_embeddings.max().item():.4e}, mean: {output_embeddings.mean().item():.4e}")

                if not outputs_are_finite:
                    logging.error("HOOK: NaN/Inf DETECTED in embedding output!")
                    nan_inf_mask_output = ~torch.isfinite(output_embeddings)
                    num_nan_inf_output = nan_inf_mask_output.sum().item()
                    logging.error(f"HOOK: Number of non-finite values in embedding output: {num_nan_inf_output} / {output_embeddings.numel()}")
                    
                    problem_indices = nan_inf_mask_output.nonzero(as_tuple=False)
                    logging.error(f"HOOK: Indices of non-finite output embeddings (first 5): {problem_indices[:5].tolist()}")

                    for i in range(min(5, problem_indices.shape[0])):
                        idx_tuple = tuple(problem_indices[i].tolist()) 
                        
                        problem_input_id_batch_idx = idx_tuple[0]
                        problem_input_id_seq_idx = idx_tuple[1]
                        
                        problem_token_id = input_ids_to_embedding[problem_input_id_batch_idx, problem_input_id_seq_idx].item()
                        problem_embedding_vector = output_embeddings[problem_input_id_batch_idx, problem_input_id_seq_idx]

                        logging.error(
                            f"HOOK: Problem case {i+1}: Input ID {problem_token_id} "
                            f"at input_ids[{problem_input_id_batch_idx}, {problem_input_id_seq_idx}] "
                            f"produced non-finite embedding vector. Vector sample (first 10): {problem_embedding_vector[:10].tolist()}"
                        )
                        
                        if 0 <= problem_token_id < module.weight.shape[0]: # Check against embedding matrix dim
                            embedding_row_for_problem_id = module.weight.data[problem_token_id]
                            is_row_finite = torch.isfinite(embedding_row_for_problem_id).all().item()
                            logging.error(
                                f"HOOK: Corresponding embedding weight row for token {problem_token_id} is finite: {is_row_finite}. "
                                f"Weight row sample: {embedding_row_for_problem_id[:10].tolist()}"
                            )
                            if is_row_finite:
                                logging.error(
                                    f"HOOK: Weight row for token {problem_token_id} - min: {embedding_row_for_problem_id.min().item():.4e}, "
                                    f"max: {embedding_row_for_problem_id.max().item():.4e}, mean: {embedding_row_for_problem_id.mean().item():.4e}"
                                )
                        else:
                            logging.error(f"HOOK: Problem token ID {problem_token_id} is out of bounds for embedding matrix lookup (vocab size {VOCAB_SIZE}, weight matrix dim0 {module.weight.shape[0]}).")

                    debug_dir = Path("./embedding_nan_debug")
                    debug_dir.mkdir(exist_ok=True)
                    torch.save(input_ids_to_embedding.cpu(), debug_dir / "problem_input_ids.pt")
                    torch.save(module.weight.data.clone().cpu(), debug_dir / "problem_embedding_weights.pt")
                    torch.save(output_embeddings.cpu(), debug_dir / "problem_output_embeddings.pt") # Save a clone of output
                    logging.info(f"HOOK: Saved debug tensors to {debug_dir}")
                    
                    nan_debug_info['triggered'] = True 
                    # nan_debug_info['hook_active'] = False # Optionally disable hook after first trigger to avoid spamming logs
                    # Consider raising an error here if you want to stop immediately after detection and saving
                    # raise RuntimeError("NaN detected in embedding output via hook, debug info saved.")
            except Exception as hook_e:
                logging.error(f"HOOK: Exception during hook execution: {hook_e}", exc_info=True)
            
        return output # Hook must return the output, or a modified one
    
    hook_handle = None
    # --- End Embedding Debug Hook ---

    try: 
        if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'embed_in'):
            hook_handle = model.gpt_neox.embed_in.register_forward_hook(embedding_forward_hook)
            logging.info("Registered forward hook on embedding layer for NaN debugging.")
        else:
            logging.warning("Could not register embedding forward hook: gpt_neox.embed_in not found.")

        # Log model parameter norm before starting the loop
        model_param_norm_before_loop = 0
        for p in model.parameters():
            model_param_norm_before_loop += p.data.norm(2).item() ** 2
        model_param_norm_before_loop = model_param_norm_before_loop ** 0.5
        logging.info(f"Model Parameter Norm before training loop: {model_param_norm_before_loop:.4f}")
        if not args.disable_wandb:
             wandb.log({ "train/model_param_norm_before_loop": model_param_norm_before_loop, "train/global_optimizer_step": 0})

        for epoch in range(args.epochs if args.epochs > 0 else 1): # Ensure at least one pass if epochs=0 for setup code
            if num_total_training_steps == 0 and args.token_budget == 0 and args.epochs == 0:
                logging.info("Skipping training loop as epochs and token_budget are 0.")
                training_complete = True
                break # Skip training loop entirely
            
            model.train()
            epoch_train_loss = 0
            num_train_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} Training", leave=False, disable=(num_total_training_steps == 0))
            for batch_idx, batch in enumerate(progress_bar):
                # This condition must be first to ensure we don't overshoot OPTIMIZER steps
                if num_total_training_steps > 0 and global_optimizer_step >= num_total_training_steps:
                    logging.info(f"Reached target global optimizer steps ({global_optimizer_step}/{num_total_training_steps}). Finishing training.")
                    training_complete = True
                    break 

                model.train() # Ensure model is in train mode 
                _first_non_finite_detected = False 
                
                input_ids = batch.to(device)
                labels = input_ids # For Causal LM, model handles shifting
                
                # --- Batch Checksum (Keep this) ---
                if not first_batch_checked: # This check is for the very first micro-batch overall
                    max_token_id = input_ids.max().item()
                    # Ensure VOCAB_SIZE is defined (loaded from tokenizer earlier)
                    assert max_token_id < VOCAB_SIZE, \
                        f"Initial batch token ID {max_token_id} >= vocab size {VOCAB_SIZE}. Check tokenization/data."
                    logging.info(f"Initial batch token ID check passed (max_id={max_token_id}, vocab_size={VOCAB_SIZE}).")
                    first_batch_checked = True
                # --- End Checksum ---
                
                # --- Gradient accumulation loop ---
                # Note: micro_step_idx goes from 0 to gradient_accumulation_steps - 1
                batch_input_ids = input_ids
                batch_target_ids = labels

                # --- Forward pass ---
                if args.precision != "fp32" and device.type == 'cuda':
                    # Autocast for bf16/fp16 on CUDA
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16):
                        outputs = model(input_ids=batch_input_ids, labels=batch_target_ids) # Pass labels for internal loss calc
                        loss = outputs.loss
                else: # fp32 or CPU (autocast context is not used for fp32)
                    outputs = model(input_ids=batch_input_ids, labels=batch_target_ids) # Pass labels for internal loss calc
                    # Check for NaN/Inf in logits before loss calculation
                    if not torch.isfinite(outputs.logits).all():
                        logging.error(f"Infinite/NaN logit detected at Opt Step {global_optimizer_step}, Micro-step {global_micro_batch_step+1} (0-indexed micro w/in accum)") # Clarified micro-step
                        logging.error(f"Logits min: {outputs.logits.min().item()}, max: {outputs.logits.max().item()}, mean: {outputs.logits.mean().item()}")
                        
                        problematic_dir = Path("./problematic_batch_data")
                        problematic_dir.mkdir(exist_ok=True)
                        torch.save(batch_input_ids, problematic_dir / f"problematic_input_ids_opt{global_optimizer_step}_micro{global_micro_batch_step+1}.pt")
                        torch.save(batch_target_ids, problematic_dir / f"problematic_target_ids_opt{global_optimizer_step}_micro{global_micro_batch_step+1}.pt")
                        # Fix for PosixPath JSON serialization
                        current_args_dict_serializable = vars(args).copy()
                        for key, value in current_args_dict_serializable.items():
                            if isinstance(value, Path):
                                current_args_dict_serializable[key] = str(value)
                        with open(problematic_dir / f"problematic_run_args_opt{global_optimizer_step}_micro{global_micro_batch_step+1}.json", "w") as f:
                            json.dump(current_args_dict_serializable, f, indent=4)
                        logging.info(f"Saved problematic batch input_ids, target_ids, and args to {problematic_dir}")

                        nan_inf_mask = ~torch.isfinite(outputs.logits)
                        num_nan_inf_logits = nan_inf_mask.sum().item()
                        logging.error(f"Number of non-finite logits: {num_nan_inf_logits} out of {outputs.logits.numel()}")
                        if num_nan_inf_logits < 20:
                            non_finite_indices = nan_inf_mask.nonzero(as_tuple=False)
                            logging.error(f"Indices of non-finite logits: {non_finite_indices.tolist()}")
                        
                        raise RuntimeError(f"Infinite/NaN logit detected at Opt Step {global_optimizer_step}, Micro-step {global_micro_batch_step+1}. See logs and problematic_batch_data/ for details.")
                    else:
                        # Logits are finite. If this is the specific micro-step of interest (Opt Step 1, Micro-step 8), log their stats.
                        # Corrected condition: Opt Step 0 (first), Micro-step gradient_accumulation_steps (e.g. 8th, so global_micro_batch_step is N-1)
                        if global_optimizer_step == 0 and global_micro_batch_step == (args.gradient_accumulation_steps -1) :
                            logits_min = outputs.logits.min().item()
                            logits_max = outputs.logits.max().item()
                            logits_mean = outputs.logits.mean().item()
                            logging.info(f"DEBUG (Opt Step {global_optimizer_step}, Micro-step about to be processed {global_micro_batch_step+1}): Finite Logits Stats before loss: Min={logits_min:.4e}, Max={logits_max:.4e}, Mean={logits_mean:.4e}")

                        loss = outputs.loss
                
                # Check loss value BEFORE backward (this check has been very useful)
                if not torch.isfinite(loss):
                    logging.error(f"NaN or Inf loss detected at Opt Step {global_optimizer_step}, Micro-step {global_micro_batch_step+1} (0-indexed) BEFORE backward(). Loss: {loss.item()}") # Clarified micro-step
                    if torch.isfinite(outputs.logits).all():
                        logging.error("Context: This NaN loss occurred despite the model's output logits being finite. Check logged logit stats if it was the problematic step.")
                    else:
                        logging.error("Context: This NaN loss likely originated from NaN/Inf logits (which should have been caught by the check above).")
                    
                    if not (Path("./problematic_batch_data") / f"problematic_input_ids_opt{global_optimizer_step}_micro{global_micro_batch_step}_loss_nan.pt").exists():
                        problematic_dir = Path("./problematic_batch_data")
                        problematic_dir.mkdir(exist_ok=True)
                        torch.save(batch_input_ids, problematic_dir / f"problematic_input_ids_opt{global_optimizer_step}_micro{global_micro_batch_step}_loss_nan.pt")
                        torch.save(batch_target_ids, problematic_dir / f"problematic_target_ids_opt{global_optimizer_step}_micro{global_micro_batch_step}_loss_nan.pt")
                        # Fix for PosixPath JSON serialization
                        current_args_dict_serializable = vars(args).copy()
                        for key, value in current_args_dict_serializable.items():
                            if isinstance(value, Path):
                                current_args_dict_serializable[key] = str(value)
                        with open(problematic_dir / f"problematic_run_args_opt{global_optimizer_step}_micro{global_micro_batch_step}_loss_nan.json", "w") as f:
                            json.dump(current_args_dict_serializable, f, indent=4)
                        logging.info(f"Saved batch data due to NaN loss (logits were finite) to {problematic_dir}")

                    raise ValueError(f"NaN or Inf loss detected BEFORE backward() at Opt Step {global_optimizer_step}, Micro-step {global_micro_batch_step}")

                loss = loss / args.gradient_accumulation_steps # Normalize loss for accumulation
                
                # --- Backward pass with scaler for mixed precision ---
                # For fp32, GradScaler is a no-op, so scaler.scale(loss) is just loss.
                # GradScaler should only be used if args.precision == 'fp16'.
                # However, the existing code uses it for bf16 as well, which is incorrect.
                # GradScaler is only for fp16. For bf16, no scaler is needed, just autocast.
                # For fp32, neither autocast nor scaler.

                # Corrected logic for backward pass based on precision:
                if args.precision == 'fp16':
                    scaler.scale(loss).backward()
                elif args.precision == 'bf16': # Autocast handles bf16, no scaler needed for backward
                    loss.backward()
                else: # fp32
                    loss.backward()
                
                accumulated_loss_for_opt_step += loss.item() 
                
                # Optimizer step check: True if (current_micro_batch_index_within_accumulation + 1) % N == 0
                # global_micro_batch_step is 0-indexed for current accumulation cycle
                if (global_micro_batch_step + 1) % args.gradient_accumulation_steps == 0:
                    
                    # Calculate norms based on accumulated gradients
                    total_norm_before_clip = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            if not torch.isfinite(p.grad.data).all():
                                logging.error(f"NaN/Inf gradient detected for param before unscale/clip at Opt Step {global_optimizer_step}, Current Micro-batch index {global_micro_batch_step}")
                                p.grad.data = torch.where(torch.isfinite(p.grad.data), p.grad.data, torch.zeros_like(p.grad.data))
                                logging.warning("Replaced non-finite gradients with zeros for safety before norm calculation.")
                            param_norm = p.grad.data.norm(2)
                            total_norm_before_clip += param_norm.item() ** 2
                    total_norm_before_clip = total_norm_before_clip ** 0.5
                    
                    # Unscale gradients before clipping if using fp16 and scaler
                    if args.precision == 'fp16':
                        scaler.unscale_(optimizer) 
                    
                    clip_threshold = args.max_grad_norm 
                    for p in model.parameters():
                        if p.grad is not None and torch.isfinite(p.grad.data).all():
                             torch.nn.utils.clip_grad_norm_(p, max_norm=clip_threshold) 
                        elif p.grad is not None: 
                             logging.warning(f"Skipping clip_grad_norm_ for a parameter with non-finite gradient at Opt Step {global_optimizer_step}")
                    
                    clip_hit = 1 if total_norm_before_clip > clip_threshold else 0

                    total_norm_after_clip = 0
                    for p in model.parameters():
                        if p.grad is not None and torch.isfinite(p.grad.data).all(): 
                            param_norm = p.grad.data.norm(2)
                            total_norm_after_clip += param_norm.item() ** 2
                        elif p.grad is not None and not torch.isfinite(p.grad.data).all():
                            logging.warning(f"Skipping norm calculation for param with non-finite grad after clip attempt at Opt Step {global_optimizer_step}")
                    total_norm_after_clip = total_norm_after_clip ** 0.5
                    
                    # Optimizer step (using scaler for fp16)
                    if args.precision == 'fp16':
                        scaler.step(optimizer)
                        scaler.update() 
                        current_grad_scaler_scale = scaler.get_scale() # Get scale factor
                    else: # bf16 or fp32
                        optimizer.step()
                        current_grad_scaler_scale = float('nan') # Not applicable

                    optimizer.zero_grad() 
                    lr_scheduler.step() # Step the scheduler *after* the optimizer step
                    
                    global_optimizer_step += 1 

                    # --- Check weights after first optimizer step (moved here) ---
                    if global_optimizer_step == 1: 
                        logging.info(f"POST OPTIMIZER STEP 0 (current global_optimizer_step={global_optimizer_step}): Checking all model weights for finiteness.")
                        all_weights_finite_after_step0 = True
                        for name, param in model.named_parameters():
                            if not torch.isfinite(param.data).all():
                                logging.error(f"POST OPTIMIZER STEP 0: Parameter '{name}' HAS NON-FINITE weights! Shape: {param.shape}, Dtype: {param.dtype}")
                                all_weights_finite_after_step0 = False
                                non_finite_values = param.data[~torch.isfinite(param.data)]
                                logging.error(f"POST OPTIMIZER STEP 0: Non-finite values in '{name}' (first 10): {non_finite_values.flatten()[:10].tolist()}")
                        if all_weights_finite_after_step0:
                            logging.info("POST OPTIMIZER STEP 0: All model weights are finite.")
                        else:
                            logging.error("POST OPTIMIZER STEP 0: NON-FINITE WEIGHTS DETECTED AFTER FIRST OPTIMIZER STEP!")
                            # torch.save(model.state_dict(), output_dir_for_run / "model_state_after_first_opt_step_nan.pt")
                            # raise RuntimeError("Non-finite weights after first optimizer step.")
                    # --- End weight check ---
                    
                    current_lr = lr_scheduler.get_last_lr()[0] if num_total_training_steps > 0 else args.learning_rate
                    current_time = time.time()
                    time_per_opt_step = current_time - last_opt_step_time
                    last_opt_step_time = current_time
                    avg_loss_this_opt_step = accumulated_loss_for_opt_step # Already divided by grad_accum_steps
                    accumulated_loss_for_opt_step = 0.0 
                    
                    # --- GEOM CKPT UPDATE ---
                    geom_ema = (avg_loss_this_opt_step if geom_ema is None
                                else beta * geom_ema + (1 - beta) * avg_loss_this_opt_step)

                    if geom_last is None or geom_ema <= alpha * geom_last:
                        # Pass necessary dependencies to save_geom_ckpt, including eval components, plot data lists, and output_dir_for_run
                        save_geom_ckpt(model, global_optimizer_step, geom_ema, loss_ckpt_dir, geom_saved, 
                                       args, wandb, logging, shutil, 
                                       val_dataloader, device, epoch, evaluate, # Eval components
                                       geom_per_token_loss_xs, geom_per_token_loss_ys_list, geom_per_token_loss_keys, # Plot data lists
                                       output_dir_for_run) # Pass the run-specific output directory
                        geom_last = geom_ema
                    # ------------------------
                    
                    if not args.disable_wandb and num_total_training_steps > 0:
                        model_param_norm_current_step = torch.nn.utils.parameters_to_vector(
                                                             [p.detach() for p in model.parameters() if torch.isfinite(p.detach()).all()] 
                                                         ).norm().item() if any(torch.isfinite(p.detach()).all() for p in model.parameters()) else float('nan')
                        
                        wandb_logs = {
                            "train/avg_loss_per_opt_step": avg_loss_this_opt_step, 
                            "train/learning_rate": current_lr,
                            "train/global_optimizer_step": global_optimizer_step,
                            "epoch": epoch + 1,
                            "train/grad_norm_before_clip": total_norm_before_clip,
                            "train/grad_norm_after_clip": total_norm_after_clip,
                            "train/clip_hit": clip_hit, 
                            "train/time_per_opt_step": time_per_opt_step, 
                            "train/model_param_norm_current_step": model_param_norm_current_step
                        }
                        if args.precision == 'fp16':
                            wandb_logs["train/grad_scaler_scale"] = current_grad_scaler_scale
                        wandb.log(wandb_logs)

                    # Console logging at log_interval (based on optimizer steps)
                    if global_optimizer_step % args.log_interval == 0 and num_total_training_steps > 0:
                        log_msg = f"Epoch {epoch+1}, Opt Step {global_optimizer_step}/{num_total_training_steps}, LR {current_lr:.2e}, Avg Loss: {avg_loss_this_opt_step:.4f}"
                        log_msg += f", Grad Norm (Before/After Clip): {total_norm_before_clip:.2f}/{total_norm_after_clip:.2f} (Hit: {clip_hit})"
                        if 'model_param_norm_current_step' in locals() and not np.isnan(model_param_norm_current_step): 
                             log_msg += f", Model Norm: {model_param_norm_current_step:.2f}"
                        if args.precision == 'fp16':
                            log_msg += f", GradScaler Scale: {current_grad_scaler_scale:.0f}"
                        log_msg += f", Time/OptStep: {time_per_opt_step:.2f}s"
                        logging.info(log_msg)
                
                    # Checkpointing based on optimizer steps
                    if args.checkpoint_interval_steps > 0 and (global_optimizer_step % args.checkpoint_interval_steps == 0) and global_optimizer_step > 0: 
                        step_checkpoint_path = step_checkpoints_dir / f"step_{global_optimizer_step}"
                        model.save_pretrained(step_checkpoint_path)
                        logging.info(f"Saved step-based checkpoint to {step_checkpoint_path} at optimizer step {global_optimizer_step}")
                        saved_step_checkpoints.append(step_checkpoint_path)
                        if args.max_step_checkpoints > 0 and len(saved_step_checkpoints) > args.max_step_checkpoints:
                            oldest_checkpoint = saved_step_checkpoints.pop(0)
                            if oldest_checkpoint.exists():
                                shutil.rmtree(oldest_checkpoint)
                                logging.info(f"Removed oldest step checkpoint: {oldest_checkpoint}")
                
                global_micro_batch_step += 1 
                if global_micro_batch_step >= args.gradient_accumulation_steps:
                    global_micro_batch_step = 0

            if training_complete: 
                break 
            
            # Perform validation at specified interval or at the end if budget not met by epoch end
            perform_eval = val_dataloader and \
                           ( (epoch + 1) % args.eval_interval == 0 or \
                             (epoch + 1) == args.epochs and not training_complete ) # Also eval at last epoch if budget not met
            
            if perform_eval:
                logging.info(f"Evaluating at end of epoch {epoch+1}...")
                # Regular epoch-end evaluation doesn't need per-token losses for this specific plot
                val_loss, _ = evaluate(model, val_dataloader, device, epoch, args, calculate_per_token_loss=False)
                logging.info(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logging.info(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                    best_model_path = output_dir_for_run / "best_model"
                    model.save_pretrained(best_model_path)
                    logging.info(f"Saved best model to {best_model_path}")
                    if not args.disable_wandb:
                        wandb.summary["best_val_loss"] = best_val_loss
            elif not val_dataloader and args.checkpoint_interval_steps == 0 and len(train_dataloader) > 0: # Check if train_dataloader is not empty 
                 logging.info(f"No validation loader and no step checkpointing. Saving model at end of epoch {epoch+1}.")
                 current_epoch_model_path = output_dir_for_run / f"model_epoch_{epoch+1}"
                 model.save_pretrained(current_epoch_model_path)
                 logging.info(f"Saved model to {current_epoch_model_path}")

    except RuntimeError as e:
        if "Function '.*' returned nan values in its 0th output." in str(e) or \
             "returned NULL output" in str(e): 
            logging.error("torch.autograd.detect_anomaly triggered! See traceback for the operation causing NaN/Inf gradients.", exc_info=True)
        else: 
             logging.exception("An unexpected RuntimeError occurred during the training loop.") 
        if not args.disable_wandb and wandb.run is not None: wandb.finish(exit_code=1) 
        raise 

    finally:
        if hook_handle: 
            hook_handle.remove()
            logging.info("Removed embedding forward hook.")
        nan_debug_info['hook_active'] = False 

        torch.autograd.set_detect_anomaly(False)
        logging.info("Disabled torch.autograd.detect_anomaly.")
        logging.info("Training loop finished or interrupted. Proceeding to final steps.")
        try:
             if 'model' in locals(): 
                 model.save_pretrained(final_saved_model_path)
                 logging.info(f"Saved final model state to {final_saved_model_path}")
             else:
                 logging.warning("Model variable not found, cannot save final model.")
        except Exception as e_save: 
             logging.error(f"Failed to save final model: {e_save}")

        if not args.disable_wandb and wandb.run is not None:
            logging.info("Finishing W&B run...")
            exit_code = 0 if training_complete or ('epoch' in locals() and epoch == args.epochs -1) else 1 
            wandb.finish(exit_code=exit_code)
            
        if args.upload_results_to_s3:
            if not args.s3_results_bucket:
                logging.error("S3 results bucket name must be provided using --s3_results_bucket to upload results.")
            else:
                s3_upload_prefix = (args.s3_results_prefix.rstrip('/') + '/' + run_name_base).lstrip('/')
                upload_directory_to_s3(output_dir_for_run, args.s3_results_bucket, s3_upload_prefix)
        
        logging.info("Script execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Pythia model on ArXiv dataset for selected categories. Model is always randomly initialized.")

    # Paths and naming
    parser.add_argument("--preprocessed_data_dir", type=Path, default=DEFAULT_PREPROCESSED_DIR, help="Directory with preprocessed data (tokens.bin, index.jsonl, splits.json).")
    parser.add_argument("--output_dir", type=Path, default=Path("./training_output"), help="Root directory to save training outputs (logs, models).")
    parser.add_argument("--run_suffix", type=str, default="", help="Optional suffix for the run name.")
    
    # Model and Tokenizer
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME, help="Model name or path for AutoConfig (e.g., EleutherAI/pythia-70m-deduped). Model will be randomly initialized using this config.")

    # Category selection
    parser.add_argument("-k", "--k", type=int, required=True, help="Number of categories to select for training (1 to 11).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for category selection and training initialization.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Maximum number of epochs. If token_budget is set, training might stop earlier. If token_budget is 0, this determines total steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per device for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate for AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1 parameter.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2 parameter.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon parameter.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum norm for gradient clipping.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type of learning rate scheduler (e.g., linear, cosine, constant).")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of warmup steps (in optimizer steps) for the LR scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of micro-batches to accumulate gradients over before performing an optimizer step.")
    parser.add_argument("--token_budget", type=int, default=0, help="Total number of tokens to train on. If >0, this primarily determines training duration. Default: 0 (use epochs).")
    parser.add_argument("--sequence_length", type=int, default=256, help="Sequence length for training samples.")

    # Dataloader and System
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2 if os.cpu_count() else 1, help="Number of worker processes for DataLoader.")
    parser.add_argument("--force_cpu", action="store_true", help="Force training on CPU even if CUDA is available.")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Training precision (default: fp32). bf16 recommended for A100 if stable.")

    # Logging and Checkpointing
    parser.add_argument("--log_interval", type=int, default=100, help="Log training loss every N optimizer steps.") 
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate on validation set every N epochs (if validation data exists).")
    parser.add_argument("--checkpoint_interval_steps", type=int, default=0, help="Save checkpoint every N optimizer steps. 0 to disable step checkpointing. Checkpoints are saved only if global_optimizer_step > 0.") 
    parser.add_argument("--max_step_checkpoints", type=int, default=3, help="Maximum number of step-based checkpoints to keep. 0 for unlimited.")

    # Geometric loss checkpoint arguments
    parser.add_argument("--geom_alpha", type=float, default=0.90, help="Save checkpoint when EMA of train loss <= alpha * last-saved EMA.")
    parser.add_argument("--geom_beta", type=float, default=0.95, help="EMA smoothing constant for geometric loss checkpointing.")
    parser.add_argument("--max_loss_ckpts", type=int, default=20, help="Maximum number of geometric loss-based checkpoints to keep. 0 for unlimited.")

    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="icl-non-ergodic-arxiv", help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (username or team).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom run name for Weights & Biases.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")

    # S3 Upload arguments
    parser.add_argument("--upload_results_to_s3", action="store_true", help="Upload final results directory to S3.")
    parser.add_argument("--s3_results_bucket", type=str, default=None, help="S3 bucket name for uploading results.")
    parser.add_argument("--s3_results_prefix", type=str, default="training_runs/", help="S3 prefix (folder) for uploading results.")
    
    args = parser.parse_args()

    if not 1 <= args.k <= len(ALL_CATEGORIES):
        parser.error(f"K must be between 1 and {len(ALL_CATEGORIES)}. Got {args.k}.")

    train(args) 