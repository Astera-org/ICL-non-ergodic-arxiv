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

def evaluate(model: AutoModelForCausalLM, dataloader: DataLoader, device: torch.device, current_epoch: int, args: argparse.Namespace) -> float:
    """Evaluates the model on the given dataloader.
    Returns average loss.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {current_epoch+1} Evaluating", leave=False):
            input_ids = batch.to(device)
            # For causal LM, the model handles shifting labels internally if `labels` are provided.
            # If labels are not provided, it computes loss against shifted input_ids.
            # Let's explicitly provide labels for clarity, same as input_ids.
            labels = input_ids 
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    if not args.disable_wandb:
        wandb.log({"eval/epoch_val_loss": avg_loss, "epoch": current_epoch + 1})
    return avg_loss

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

    # Enable anomaly detection for debugging NaNs
    torch.autograd.set_detect_anomaly(True)

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

    # 3. Load Model and Tokenizer
    logging.info(f"Loading model and tokenizer: {args.model_name_or_path}")
    logging.info(f"Initializing model from config: {args.model_name_or_path} (random initialization).")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_config(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}")
    model.to(device)

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
                                eps=1e-8, # Reverted to default/requested
                                betas=(0.9, 0.95)) # Use requested betas
    
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
    global_micro_batch_step = 0
    best_val_loss = float('inf')
    saved_step_checkpoints = []
    training_complete = False
    first_batch_checked = False # Flag for vocab check
    accumulated_loss_for_opt_step = 0.0 # Accumulator for average loss
    last_opt_step_time = time.time() # For timing steps
    scaler = GradScaler(enabled=(args.precision == 'fp16')) # GradScaler for fp16
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32 # Determine autocast dtype
    final_saved_model_path = output_dir_for_run / "final_model" # Define path earlier

    try: 
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
                input_ids = batch.to(device)
                labels = input_ids # For Causal LM, model handles shifting
                
                # --- Batch Checksum (Run Once) ---
                if not first_batch_checked:
                    max_token_id = input_ids.max().item()
                    assert max_token_id < VOCAB_SIZE, \
                        f"Batch contains token ID {max_token_id} >= vocab size {VOCAB_SIZE}. Check tokenization/data."
                    logging.info("First batch token ID check passed.")
                    first_batch_checked = True
                # --- End Checksum ---
                
                # Mixed Precision Context
                with autocast(enabled=(args.precision != 'fp32'), dtype=dtype):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"NaN or Inf loss detected at Micro-step {global_micro_batch_step}, Opt Step {global_optimizer_step}, Epoch {epoch+1}. Loss: {loss.item()}")
                    # Optionally, log the input_ids that caused the NaN/Inf loss
                    # logging.error(f"Problematic input_ids (first example in batch): {input_ids[0][:20]}...") 
                    # Consider breaking or handling appropriately
                    if not args.disable_wandb and wandb.run is not None: wandb.finish(exit_code=1)
                    raise ValueError(f"NaN or Inf loss detected at Opt Step {global_optimizer_step}")

                # Normalize loss for accumulation
                normalized_loss = loss / args.gradient_accumulation_steps
                
                # Scale loss and call backward using GradScaler for fp16, normal backward for bf16/fp32
                scaler.scale(normalized_loss).backward()
                
                accumulated_loss_for_opt_step += loss.item() # Accumulate original loss
                
                # Log micro-batch loss (optional, but can be useful)
                if not args.disable_wandb and num_total_training_steps > 0:
                     wandb.log({ "train/micro_batch_loss": loss.item(), # Log original loss
                                 "train/global_micro_batch_step": global_micro_batch_step})
                
                global_micro_batch_step += 1

                # Optimizer step check
                if global_micro_batch_step % args.gradient_accumulation_steps == 0:
                    global_optimizer_step += 1
                    
                    # Calculate norms based on accumulated gradients
                    total_norm_before_clip = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm_before_clip += param_norm.item() ** 2
                    total_norm_before_clip = total_norm_before_clip ** 0.5
                    
                    # Add gradient clipping (Unscale grads first for fp16)
                    scaler.unscale_(optimizer) # Unscale gradients before clipping for fp16
                    clip_threshold = args.max_grad_norm # Use arg for max_grad_norm
                    clip_hit = 1 if total_norm_before_clip > clip_threshold else 0
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_threshold) # Clip accumulated gradients

                    total_norm_after_clip = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm_after_clip += param_norm.item() ** 2
                    total_norm_after_clip = total_norm_after_clip ** 0.5
                    
                    # Optimizer step (using scaler for fp16)
                    scaler.step(optimizer)
                    scaler.update() # Update scaler for next iteration (for fp16)
                    
                    current_lr = lr_scheduler.get_last_lr()[0] if num_total_training_steps > 0 else args.learning_rate
                    
                    # --- Timing --- 
                    current_time = time.time()
                    time_per_opt_step = current_time - last_opt_step_time
                    last_opt_step_time = current_time
                    
                    # --- Calculate Average Loss for Opt Step --- 
                    avg_loss_this_opt_step = accumulated_loss_for_opt_step / args.gradient_accumulation_steps
                    accumulated_loss_for_opt_step = 0.0 # Reset accumulator
                    
                    # --- Logging per Optimizer Step ---
                    # W&B logging every optimizer step
                    if not args.disable_wandb and num_total_training_steps > 0:
                        # Use more efficient parameter norm calculation
                        model_param_norm_current_step = torch.nn.utils.parameters_to_vector(
                                                             [p.detach() for p in model.parameters()]
                                                         ).norm().item()
                        
                        wandb_logs = {
                            "train/avg_loss_per_opt_step": avg_loss_this_opt_step, 
                            "train/learning_rate": current_lr,
                            "train/global_optimizer_step": global_optimizer_step,
                            "epoch": epoch + 1,
                            "train/grad_norm_before_clip": total_norm_before_clip,
                            "train/grad_norm_after_clip": total_norm_after_clip,
                            "train/clip_hit": clip_hit, # Log if clipping occurred
                            "train/time_per_opt_step": time_per_opt_step, # Log step time
                            "train/model_param_norm_current_step": model_param_norm_current_step
                        }
                        wandb.log(wandb_logs)

                    # Console logging at log_interval (based on optimizer steps)
                    if global_optimizer_step % args.log_interval == 0 and num_total_training_steps > 0:
                        logging.info(f"Epoch {epoch+1}, Opt Step {global_optimizer_step}/{num_total_training_steps}, LR {current_lr:.2e}, Avg Loss: {avg_loss_this_opt_step:.4f}") 
                        logging.info(f"Gradient Norm (Opt Step): Before Clip={total_norm_before_clip:.4f}, After Clip={total_norm_after_clip:.4f} (Clip Hit: {clip_hit})")
                        if 'model_param_norm_current_step' in locals(): 
                             logging.info(f"Model Parameter Norm (Opt Step): {model_param_norm_current_step:.4f}")
                        logging.info(f"Time per Opt Step: {time_per_opt_step:.2f}s")
                
                    # Checkpointing based on optimizer steps
                    if args.checkpoint_interval_steps > 0 and (global_micro_batch_step % args.gradient_accumulation_steps == 0) and (global_optimizer_step % args.checkpoint_interval_steps == 0) and global_optimizer_step > 0:
                        step_checkpoint_path = step_checkpoints_dir / f"step_{global_optimizer_step}"
                        model.save_pretrained(step_checkpoint_path)
                        logging.info(f"Saved step-based checkpoint to {step_checkpoint_path} at optimizer step {global_optimizer_step}")
                        saved_step_checkpoints.append(step_checkpoint_path)
                        if args.max_step_checkpoints > 0 and len(saved_step_checkpoints) > args.max_step_checkpoints:
                            oldest_checkpoint = saved_step_checkpoints.pop(0)
                            if oldest_checkpoint.exists():
                                shutil.rmtree(oldest_checkpoint)
                                logging.info(f"Removed oldest step checkpoint: {oldest_checkpoint}")
            
            if training_complete: 
                break 
            
            # Log epoch average loss only if some training happened in this epoch
            # Note: epoch_train_loss accumulates the *last* micro-batch loss, not the average
            # This calculation might be less meaningful now. We could accumulate avg_loss_this_opt_step instead?
            # Let's comment it out for now to avoid confusion, as optimizer-step avg loss is logged.
            # if num_train_batches > 0: 
            #     avg_epoch_train_loss = epoch_train_loss / num_train_batches
            #     logging.info(f"Epoch {epoch+1} average training loss: {avg_epoch_train_loss:.4f}")
            #     if not args.disable_wandb:
            #         wandb.log({"train/epoch_avg_loss": avg_epoch_train_loss, "epoch": epoch + 1})

            # Perform validation at specified interval or at the end if budget not met by epoch end
            perform_eval = val_dataloader and \
                           ( (epoch + 1) % args.eval_interval == 0 or \
                             (epoch + 1) == args.epochs and not training_complete ) # Also eval at last epoch if budget not met
            
            if perform_eval:
                logging.info(f"Evaluating at end of epoch {epoch+1}...")
                val_loss = evaluate(model, val_dataloader, device, epoch, args)
                logging.info(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logging.info(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                    best_model_path = output_dir_for_run / "best_model"
                    model.save_pretrained(best_model_path)
                    # If a tokenizer were used and needed saving: tokenizer.save_pretrained(best_model_path)
                    logging.info(f"Saved best model to {best_model_path}")
                    if not args.disable_wandb:
                        wandb.summary["best_val_loss"] = best_val_loss
            elif not val_dataloader and args.checkpoint_interval_steps == 0 and num_train_batches > 0: 
                 logging.info(f"No validation loader and no step checkpointing. Saving model at end of epoch {epoch+1}.")
                 current_epoch_model_path = output_dir_for_run / f"model_epoch_{epoch+1}"
                 model.save_pretrained(current_epoch_model_path)
                 logging.info(f"Saved model to {current_epoch_model_path}")

    except Exception as e:
        logging.exception("An error occurred during the training loop.")
        # Ensure wandb is finished even if error occurs
        if not args.disable_wandb and wandb.run is not None:
            wandb.finish(exit_code=1) 
        # Don't re-raise yet, allow finally block to run
        raise # Re-raise after finally

    finally:
        logging.info("Training loop finished or interrupted. Proceeding to final steps.")
        try:
             # Save final model state regardless of loop completion reason (unless error stopped before model init)
             if 'model' in locals(): 
                 model.save_pretrained(final_saved_model_path)
                 logging.info(f"Saved final model state to {final_saved_model_path}")
             else:
                 logging.warning("Model variable not found, cannot save final model.")
        except Exception as e:
             logging.error(f"Failed to save final model: {e}")

        if not args.disable_wandb and wandb.run is not None:
            logging.info("Finishing W&B run...")
            exit_code = 0 if training_complete or epoch == args.epochs -1 else 1 # Mark successful if loop completed naturally or hit budget
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

    # Training hyperparameters (Original location)
    parser.add_argument("--epochs", type=int, default=3, help="Maximum number of epochs. If token_budget is set, training might stop earlier. If token_budget is 0, this determines total steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per device for training.") # Renamed help text
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate for AdamW optimizer.") # Updated help text
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type of learning rate scheduler (e.g., linear, cosine, constant).") # Updated help text
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of warmup steps (in optimizer steps) for the LR scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of micro-batches to accumulate gradients over before performing an optimizer step.")
    parser.add_argument("--token_budget", type=int, default=0, help="Total number of tokens to train on. If >0, this primarily determines training duration. Default: 0 (use epochs).") # Added token budget here

    # Dataloader and System
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2 if os.cpu_count() else 1, help="Number of worker processes for DataLoader.")
    parser.add_argument("--force_cpu", action="store_true", help="Force training on CPU even if CUDA is available.")

    # Logging and Checkpointing (add more later)
    parser.add_argument("--log_interval", type=int, default=100, help="Log training loss every N steps.")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate on validation set every N epochs (if validation data exists).")

    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="icl-non-ergodic-arxiv", help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (username or team).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom run name for Weights & Biases.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")

    # New arguments for step-based checkpointing
    parser.add_argument("--checkpoint_interval_steps", type=int, default=0, help="Save checkpoint every N steps. 0 to disable step checkpointing. Checkpoints are saved only if global_step > 0.")
    parser.add_argument("--max_step_checkpoints", type=int, default=3, help="Maximum number of step-based checkpoints to keep. 0 for unlimited.")

    # S3 Upload arguments
    parser.add_argument("--upload_results_to_s3", action="store_true", help="Upload final results directory to S3.")
    parser.add_argument("--s3_results_bucket", type=str, default=None, help="S3 bucket name for uploading results.")
    parser.add_argument("--s3_results_prefix", type=str, default="training_runs/", help="S3 prefix (folder) for uploading results.")

    # --- Training Configuration --- (Newer block, likely source of duplication)
    parser.add_argument("--sequence_length", type=int, default=256, help="Sequence length for training samples.")
    # Remove duplicate --epochs definition from here
    # parser.add_argument("--epochs", type=int, default=3, help="Maximum number of epochs. If token_budget is set, training might stop earlier. If token_budget is 0, this determines total steps.") 
    # Remove duplicate --batch_size definition (already above)
    # parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per device for training.") 
    # Remove duplicate --learning_rate definition (already above)
    # parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate for AdamW optimizer.") 
    # Remove duplicate --weight_decay definition (already above)
    # parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.") 
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1 parameter.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2 parameter.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon parameter.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum norm for gradient clipping.")
    # Remove duplicate --lr_scheduler_type definition (already above)
    # parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type of learning rate scheduler (e.g., linear, cosine, constant).") 
    # Remove duplicate --num_warmup_steps definition (already above)
    # parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of warmup steps (in optimizer steps) for the LR scheduler.") 
    # Remove duplicate --gradient_accumulation_steps definition (already above)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of micro-batches to accumulate gradients over before performing an optimizer step.") 
    
    # --- Precision --- 
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Training precision (bf16 recommended for A100).")
    
    # Dataloader and System
    # parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2 if os.cpu_count() else 1, help="Number of worker processes for DataLoader.")
    # parser.add_argument("--force_cpu", action="store_true", help="Force training on CPU even if CUDA is available.")

    args = parser.parse_args()

    if not 1 <= args.k <= len(ALL_CATEGORIES):
        parser.error(f"K must be between 1 and {len(ALL_CATEGORIES)}. Got {args.k}.")

    train(args) 