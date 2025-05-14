import hydra
from omegaconf import DictConfig, OmegaConf
import os
import wandb # Import wandb
import torch # For torch.optim and clip_grad_norm
from torch.utils.data import DataLoader, random_split, Dataset # Keep Dataset for type hinting if needed
from tqdm import tqdm # For progress bars
from transformers import PreTrainedTokenizerFast, get_scheduler # AdamW removed from here
from torch.optim import AdamW # AdamW imported from torch.optim
from pathlib import Path
import torch.cuda.amp as amp # For mixed precision
import math # For perplexity calculation

from .logging_config import setup_logging, get_logger
from .utils import set_seed # Import set_seed
from .tokenizer_utils import load_tokenizer_from_config, chunk_token_ids # Keep load_tokenizer_from_config
from .models.model_loading import load_micro_decoder_from_config # Keep
from .window_loader import HDF5WindowLoader # Add this import
from .checkpoint_utils import CheckpointManager # Add this import

# Adjusted to be relative if logging_config is in the same directory or a known path
# If src is in PYTHONPATH, `from logging_config import ...` might work directly
# Otherwise, use relative imports if this script is part of a package: `from .logging_config ...`

# Helper function to calculate total training steps
def calculate_total_training_steps(num_epochs: int, num_batches_per_epoch: int, grad_accum_steps: int) -> int:
    """Calculates the total number of optimizer steps for the training run."""
    if grad_accum_steps <= 0:
        raise ValueError("Gradient accumulation steps must be positive.")
    steps_per_epoch = num_batches_per_epoch // grad_accum_steps
    # Add an extra step if there's a partial batch at the end of an epoch due to accumulation
    # This is a bit subtle: if num_batches_per_epoch is not a multiple of grad_accum_steps,
    # the last few batches in an epoch might not trigger an optimizer step if not handled.
    # However, the loop structure usually handles this by processing remaining gradients.
    # A simpler approach is just steps_per_epoch * num_epochs if we assume a full final accumulation step if needed.
    # For schedulers, it's often safer to slightly overestimate than underestimate if unsure.
    # Let's stick to the core logic: number of optimizer steps.
    return steps_per_epoch * num_epochs

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main_app(cfg: DictConfig) -> None:
    # Setup logging first
    setup_logging(cfg=cfg)
    log = get_logger(__name__)

    # Set seed for reproducibility as early as possible
    set_seed(cfg.seed)
    log.info(f"Global random seed set to: {cfg.seed}")

    # Initialize W&B
    wandb_run = None # Initialize wandb_run to None
    try:
        # Allow WANDB_MODE=disabled for offline runs or testing
        if os.environ.get("WANDB_MODE", "online") == "disabled" or cfg.get("wandb", {}).get("mode", "online") == "disabled":
            log.info("W&B is disabled via WANDB_MODE or hydra config.")
            # wandb_run remains None, which is fine
        else:
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity if cfg.wandb.entity else None, # Handles empty string for default entity
                name=cfg.wandb.run_name,
                tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
                config=OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False),
                notes=cfg.wandb.get("notes", None), 
                mode=cfg.wandb.get("mode", "online") # online, offline, disabled
            )
            if wandb_run and cfg.wandb.log_code:
                # Log current git state if available
                # wandb.log_code(".") # requires wandb >= 0.12.10, logs entire directory
                pass # Placeholder for more specific code logging if needed
            log.info(f"W&B run initialized: {wandb_run.name if wandb_run else 'None'}")
            log.info(f"W&B project: {cfg.wandb.project}, entity: {cfg.wandb.entity or 'default'}")

    except Exception as e: # Catches any exception during wandb.init
        log.error(f"Error initializing W&B: {e}. Setting W&B to disabled mode.", exc_info=True)
        # wandb_run remains None, effectively disabling W&B for this run.
        # No need to explicitly set mode to "disabled" here if wandb_run is None and checked later.

    log.info("Hydra application started!")
    log.info(f"Project Name: {cfg.project_name}")
    log.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Example of accessing nested config
    log.info(f"Dataset Path: {cfg.dataset.path}")
    log.info(f"Model Name: {cfg.model.name}")
    log.info(f"Training Epochs: {cfg.training.epochs}")

    # --- Accessing Experiment Configuration ---
    print("\n--- Experiment Configuration ---")
    # The experiment config is now part of the main `cfg` object under the `experiment` key
    # e.g., cfg.experiment.k_value, cfg.experiment.active_categories
    if hasattr(cfg, 'experiment'):
        print(f"Experiment ID: {cfg.experiment.experiment_id}")
        print(f"K Value: {cfg.experiment.k_value}")
        print(f"Active Categories: {cfg.experiment.active_categories}")
        print(f"Tokens per Category: {cfg.experiment.tokens_per_category}")
        
        # You can also load it into the Pydantic model if needed for validation or methods
        try:
            loaded_exp_config = ExperimentConfig(**OmegaConf.to_container(cfg.experiment, resolve=True))
            print("Successfully loaded experiment config into Pydantic model:")
            print(loaded_exp_config)
        except Exception as e:
            print(f"Could not load experiment config into Pydantic model: {e}")
    else:
        print("No experiment configuration found in Hydra config.")
    # --- End Experiment Configuration ---

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- 1. Load Model ---
    log.info(f"Loading model: {cfg.model.name}")
    tokenizer: PreTrainedTokenizerFast | None = None # To store loaded tokenizer for pad_token_id
    try:
        # First, load tokenizer to get pad_token_id (needed for loss) and vocab_size for model
        # The model_cfg (cfg.model) should contain custom_tokenizer_path
        tokenizer = load_tokenizer_from_config(cfg.model)
        if tokenizer is None:
            raise ValueError("Tokenizer could not be loaded.")
        log.info(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}, Pad token ID: {tokenizer.pad_token_id}")
        
        # Dynamically set vocab_size in model config if not present or to confirm
        # This assumes cfg.model.architecture is the right place.
        # Create a mutable copy for modification
        model_arch_cfg = OmegaConf.to_container(cfg.model.architecture, resolve=True)
        model_arch_cfg['vocab_size'] = tokenizer.get_vocab_size()
        
        # If your load_micro_decoder_from_config expects cfg.model.architecture to be a DictConfig:
        cfg.model.architecture = OmegaConf.create(model_arch_cfg)

        model = load_micro_decoder_from_config(model_cfg=cfg.model, global_cfg=cfg)
        model.to(device) # Move model to device
        log.info(f"Model '{cfg.model.name}' loaded successfully and moved to {device}.")
        # Log model parameter count
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Model total parameters: {num_params:,}")
        log.info(f"Model trainable parameters: {num_trainable_params:,}")
        if wandb_run:
            wandb.summary["model_total_parameters"] = num_params
            wandb.summary["model_trainable_parameters"] = num_trainable_params
    except Exception as e:
        log.error(f"Error loading model or tokenizer: {e}", exc_info=True)
        if wandb_run: # wandb_run might not be defined if setup_logging is not called yet
             if 'wandb_run' in locals() and wandb_run:  # Check if wandb_run is defined and not None
                wandb.finish(exit_code=1)
        return # Exit if model loading fails
    
    if tokenizer is None or tokenizer.pad_token_id is None:
        log.error("Tokenizer or pad_token_id is None after model loading. Exiting.")
        if 'wandb_run' in locals() and wandb_run:
            wandb.finish(exit_code=1)
        return

    # --- 2. Dataset Setup & Training Parameter Calculation ---
    log.info("Loading HDF5 dataset using HDF5WindowLoader...")
    try:
        # Ensure active_categories are available from the experiment config
        if not hasattr(cfg, 'experiment') or not hasattr(cfg.experiment, 'active_categories'):
            log.error("Experiment configuration for 'active_categories' not found (expected at cfg.experiment.active_categories). Exiting.")
            if wandb_run: wandb.finish(exit_code=1)
            return
        
        active_categories = list(cfg.experiment.active_categories)
        if not active_categories:
            log.error("'active_categories' list is empty in the experiment configuration. Exiting.")
            if wandb_run: wandb.finish(exit_code=1)
            return

        log.info(f"Using active categories for HDF5WindowLoader: {active_categories}")

        # Instantiate HDF5WindowLoader for training
        # It requires cfg (for dataset path), active_categories, and a seed.
        train_dataset = HDF5WindowLoader(
            cfg=cfg, 
            active_categories=active_categories, 
            seed=cfg.seed
        )
        log.info(f"Dataset: {cfg.dataset.path}, Active Categories: {active_categories}, Num Chunks: {len(train_dataset)}")
        log.info(f"An epoch will process {len(train_dataset)} chunks, totaling {len(train_dataset) * train_dataset.token_chunk_size:,} tokens.")
        if wandb_run:
            wandb.config.update({
                "dataset_active_categories": active_categories,
                "dataset_num_chunks_per_epoch": len(train_dataset),
                "dataset_tokens_per_epoch": len(train_dataset) * train_dataset.token_chunk_size
            })

    except Exception as e:
        log.error(f"Failed to load HDF5 dataset: {e}", exc_info=True)
        if wandb_run: wandb.finish(exit_code=1)
        return

    # At this point, train_dataset contains tensors of input_ids.
    # No further tokenization or .map() is needed.
    # The HDF5SequentialDataset should already return {'input_ids': torch.tensor(...)}

    # --- DataLoader ---
    # Removed custom collate_fn as it's not strictly necessary for fixed-length pre-chunked data.
    # The default collate_fn should stack the tensors correctly.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True, 
        num_workers=cfg.training.get("num_workers", 0), # Default to 0 if not specified
        pin_memory=cfg.training.get("pin_memory", True)   # Default to True if not specified
    )
    log.info(f"Train DataLoader initialized. Num batches: {len(train_dataloader)}")

    val_dataloader = None
    if cfg.training.perform_validation:
        log.info("Setting up validation dataset and DataLoader...")
        try:
            # Assume validation uses the same active_categories as training for now
            # If different categories are needed for validation, this logic will need adjustment
            # and a new config option (e.g., cfg.validation.active_categories)
            validation_active_categories = list(cfg.experiment.active_categories) 
            if not validation_active_categories:
                log.warning("'active_categories' for validation is empty. Validation may not be meaningful.")
            
            log.info(f"Using active categories for Validation HDF5WindowLoader: {validation_active_categories}")
            
            # Create a new cfg for validation dataset, overriding the path
            val_dataset_cfg = OmegaConf.create({
                "dataset": {
                    "hdf5_chunked_output_path": cfg.dataset.validation_hdf5_path,
                    # Potentially other dataset params if they differ for validation
                }
            })
            # Merge with the main cfg to ensure other necessary global settings are available if HDF5WindowLoader needs them
            # Be careful with merging if keys overlap unintentionally. Here, HDF5WindowLoader mainly needs dataset.hdf5_chunked_output_path from its direct cfg arg.
            # It might be cleaner if HDF5WindowLoader took path directly instead of full cfg.
            # For now, we construct a minimal cfg for it.

            val_dataset = HDF5WindowLoader(
                cfg=val_dataset_cfg, # Pass the minimal config with the correct path
                active_categories=validation_active_categories,
                seed=cfg.seed + 1 # Use a different seed for validation data shuffling if any internal shuffling were added to loader
            )
            log.info(f"Validation Dataset: {cfg.dataset.validation_hdf5_path}, Active Categories: {validation_active_categories}, Num Chunks: {len(val_dataset)}")
            log.info(f"A validation epoch will process {len(val_dataset)} chunks, totaling {len(val_dataset) * val_dataset.token_chunk_size:,} tokens.")
            if wandb_run:
                wandb.config.update({
                    "validation_dataset_path": cfg.dataset.validation_hdf5_path,
                    "validation_dataset_active_categories": validation_active_categories,
                    "validation_dataset_num_chunks_per_epoch": len(val_dataset),
                    "validation_dataset_tokens_per_epoch": len(val_dataset) * val_dataset.token_chunk_size
                })

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.training.validation_batch_size,
                shuffle=False, # No need to shuffle validation data
                num_workers=cfg.training.get("num_workers", 0),
                pin_memory=cfg.training.get("pin_memory", True)
            )
            log.info(f"Validation DataLoader initialized. Num batches: {len(val_dataloader)}")

        except FileNotFoundError:
            log.error(f"Validation HDF5 file not found at {cfg.dataset.validation_hdf5_path}. Disabling validation.")
            cfg.training.perform_validation = False # Disable if file not found
            val_dataloader = None
        except Exception as e:
            log.error(f"Failed to load validation HDF5 dataset: {e}. Disabling validation.", exc_info=True)
            cfg.training.perform_validation = False # Disable on other errors
            val_dataloader = None
    else:
        log.info("Validation is disabled via cfg.training.perform_validation.")

    # --- Calculate Gradient Accumulation Steps ---
    if cfg.training.batch_size > 0 and cfg.training.effective_batch_size > 0:
        calculated_grad_accum_steps = cfg.training.effective_batch_size // cfg.training.batch_size
        if calculated_grad_accum_steps <= 0: # Should not happen if effective >= batch_size
            log.warning(f"Calculated grad_accum_steps ({calculated_grad_accum_steps}) is not positive. Defaulting to 1. Check batch_size and effective_batch_size.")
            calculated_grad_accum_steps = 1
        # Update the cfg object directly if it's mutable, or use a variable
        # For simplicity, let's assume cfg.training is mutable or we re-assign if needed for clarity.
        # OmegaConf allows direct modification of loaded DictConfig objects.
        cfg.training.gradient_accumulation_steps = calculated_grad_accum_steps
        log.info(f"Calculated and set gradient_accumulation_steps: {cfg.training.gradient_accumulation_steps}")
    else:
        log.warning("Batch size or effective batch size is not positive. Using placeholder gradient_accumulation_steps: {cfg.training.gradient_accumulation_steps}")
        # Ensure it's at least 1 if calculation failed
        if cfg.training.gradient_accumulation_steps <= 0:
            cfg.training.gradient_accumulation_steps = 1

    # --- 3. Instantiate Optimizer ---
    log.info(f"Instantiating optimizer: {cfg.optimizer._target_}")
    log.info(f"Optimizer LR: {cfg.optimizer.lr}") 
    if hasattr(cfg.training, 'learning_rate') and cfg.training.learning_rate != cfg.optimizer.lr:
        log.warning(f"Training config LR ({cfg.training.learning_rate}) differs from Optimizer LR ({cfg.optimizer.lr}). Using Optimizer LR from optimizer config.")
    
    try:
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        log.info("Optimizer instantiated successfully.")
    except Exception as e:
        log.error(f"Error instantiating optimizer: {e}", exc_info=True)
        if 'wandb_run' in locals() and wandb_run: wandb.finish(exit_code=1)
        return

    # --- 4. Instantiate Scheduler ---
    log.info(f"Instantiating scheduler: {cfg.scheduler._target_}")
    scheduler_cfg_dict = OmegaConf.to_container(cfg.scheduler, resolve=True) # Use a different var name
    scheduler_cfg_dict["num_training_steps"] = calculate_total_training_steps(
        num_epochs=cfg.training.epochs,
        num_batches_per_epoch=len(train_dataloader),
        grad_accum_steps=cfg.training.gradient_accumulation_steps
    )
    if scheduler_cfg_dict["num_warmup_steps"] > scheduler_cfg_dict["num_training_steps"]:
        new_warmup_steps = min(scheduler_cfg_dict["num_warmup_steps"], scheduler_cfg_dict["num_training_steps"]) # Ensure it's not negative if num_training_steps is very small
        log.warning(f"Warmup steps ({scheduler_cfg_dict['num_warmup_steps']}) > Total training steps ({scheduler_cfg_dict['num_training_steps']}). Adjusting warmup steps to {new_warmup_steps}.")
        scheduler_cfg_dict["num_warmup_steps"] = new_warmup_steps
    
    log.info(f"Scheduler num_warmup_steps: {scheduler_cfg_dict['num_warmup_steps']}")
    log.info(f"Scheduler num_training_steps: {scheduler_cfg_dict['num_training_steps']}")

    try:
        # Remove keys not expected by the target Hugging Face scheduler function
        # min_lr_ratio is a custom informational key in our YAML, not a direct arg for get_cosine_schedule_with_warmup
        scheduler_cfg_dict.pop("min_lr_ratio", None) # Safely remove if it exists

        # Use Hydra to instantiate the scheduler using its _target_ and resolved parameters
        lr_scheduler = hydra.utils.instantiate(scheduler_cfg_dict, optimizer=optimizer)
        log.info("Learning rate scheduler instantiated successfully via Hydra.")
    except Exception as e:
        log.error(f"Error instantiating scheduler: {e}", exc_info=True)
        if 'wandb_run' in locals() and wandb_run: wandb.finish(exit_code=1)
        return

    # --- 3.5 Initialize Checkpoint Manager & Resume Logic ---
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_manager = CheckpointManager(cfg=cfg, hydra_run_dir=hydra_output_dir)
    
    start_epoch = 0
    global_step = 0
    
    resume_checkpoint_path_str = cfg.training.get("resume_from_checkpoint", None)
    load_artifact_name_for_wandb = None # For W&B artifact loading

    if resume_checkpoint_path_str:
        log.info(f"Attempting to resume from checkpoint: {resume_checkpoint_path_str}")
        
        # Determine if it's a W&B artifact URI or a local path
        is_wandb_uri = ":" in resume_checkpoint_path_str and ("/" in resume_checkpoint_path_str.split(":")[0]) # Simple check
        
        path_to_load = resume_checkpoint_path_str # Default to the string itself

        if is_wandb_uri:
            load_artifact_name_for_wandb = resume_checkpoint_path_str
            # If W&B loading were enabled, filename hint for artifact would be 'latest_checkpoint.pt' or parsed.
            # Since W&B artifact loading is disabled in CheckpointManager, this mainly serves as a flag.
            path_to_load = "latest_checkpoint.pt" # Placeholder, as actual W&B loading is off
            log.info(f"W&B artifact URI detected: {load_artifact_name_for_wandb}. (Note: W&B artifact loading is disabled in CheckpointManager)")
        else:
            # It's a local path. Resolve it to an absolute path from project root.
            # Assume hydra_output_dir is available and gives us a path relative to which we can resolve,
            # or simply use Path.cwd() if paths are from project root.
            # For paths given in config like "outputs/runs/...", they are relative to project root.
            project_root = Path(hydra.utils.get_original_cwd()) # Get original CWD where Hydra was launched
            absolute_path_to_load = project_root / resume_checkpoint_path_str
            if absolute_path_to_load.exists():
                path_to_load = str(absolute_path_to_load.resolve())
                log.info(f"Resolved local checkpoint path to absolute: {path_to_load}")
            else:
                log.warning(f"Specified local checkpoint path {absolute_path_to_load} does not exist. Will attempt to load as relative from current run's checkpoint dir.")
                # path_to_load remains resume_checkpoint_path_str (which could be e.g. "latest.pt")
                # This case might be problematic if user intended a specific old run and it's not found.

        # The load_checkpoint in CheckpointManager now expects an absolute path or a simple filename.
        # If W&B loading was active, load_from_wandb_artifact would be used.
        # Since it's disabled, we rely on path_to_load being correctly an absolute path to the target checkpoint,
        # or a simple filename if we intended to load from the *current* run's dir (which isn't the case for resuming an *old* run).
        checkpoint_data = checkpoint_manager.load_checkpoint(
            model, optimizer, lr_scheduler, 
            filename=path_to_load, # This will now be an absolute path if resolved, or original string
            device=device,
            load_from_wandb_artifact=None # W&B artifact loading is disabled in CheckpointManager
        )
        if checkpoint_data:
            start_epoch, global_step, loaded_metrics, _ = checkpoint_data
            log.info(f"Resumed from checkpoint. Epoch: {start_epoch}, Global_step: {global_step}, Metrics: {loaded_metrics}")
        else:
            log.warning(f"Failed to load checkpoint from '{resume_checkpoint_path_str}'. Starting training from scratch.")
    else:
        log.info("No checkpoint specified for resumption. Starting training from scratch.")

    # --- 5. Training Loop ---
    log.info("Starting training loop...")
    model.train() # Set model to training mode

    # Initialize GradScaler if mixed precision is enabled and CUDA is available
    scaler = None
    use_amp = cfg.training.mixed_precision_enabled and device.type == 'cuda'
    if use_amp:
        amp_dtype_str = cfg.training.mixed_precision_dtype.lower()
        if amp_dtype_str == "fp16":
            amp_dtype = torch.float16
        elif amp_dtype_str == "bf16":
            amp_dtype = torch.bfloat16
        else:
            log.warning(f"Unsupported mixed_precision_dtype: {cfg.training.mixed_precision_dtype}. Defaulting to float16 for AMP.")
            amp_dtype = torch.float16 # Default if unspecified or wrong
        scaler = amp.GradScaler()
        log.info(f"Mixed precision training enabled with dtype: {amp_dtype_str}")
    else:
        log.info("Mixed precision training disabled or CUDA not available.")

    # Get max_global_steps from config, default to -1 (no limit) if not present
    max_global_steps = cfg.training.get("max_global_steps", -1)
    if max_global_steps > 0:
        log.info(f"Training will be limited to a maximum of {max_global_steps} global steps.")

    # --- Early Stopping & Best Metric Tracking ---
    best_val_metric = float('inf') if cfg.training.save_best_model_mode == "min" else float('-inf')
    epochs_without_improvement = 0

    training_should_continue = True # Flag to break outer loop
    current_epoch_for_save = start_epoch # Initialize with start_epoch for final save robustness

    for epoch in range(start_epoch, cfg.training.epochs):
        current_epoch_for_save = epoch # Update it as the loop progresses
        if not training_should_continue:
            break
        log.info(f"--- Epoch {epoch+1}/{cfg.training.epochs} ---")
        optimizer.zero_grad() # Clear gradients at the start of each epoch accumulation cycle
        total_loss = 0.0 # Initialize total_loss for the current epoch

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            
            # Mixed precision context
            if use_amp:
                with amp.autocast(device_type=device.type, dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs[0] / cfg.training.gradient_accumulation_steps
            else:
                outputs = model(input_ids=input_ids, labels=input_ids) 
                loss = outputs[0] / cfg.training.gradient_accumulation_steps 
            
            if use_amp and scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * cfg.training.gradient_accumulation_steps # Accumulate pre-scaling loss

            if (batch_idx + 1) % cfg.training.gradient_accumulation_steps == 0:
                if use_amp and scaler:
                    if cfg.training.max_grad_norm > 0:
                        scaler.unscale_(optimizer) # Unscale gradients before clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if cfg.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad() 
                global_step += 1

                # Logging
                current_lr = lr_scheduler.get_last_lr()[0]
                avg_loss_so_far = total_loss / (batch_idx + 1) 
                
                if (batch_idx + 1) % (cfg.training.log_interval * cfg.training.gradient_accumulation_steps) == 0:
                    log.info(f"  Batch {batch_idx+1}/{len(train_dataloader)}, GlobalStep: {global_step}, LR: {current_lr:.2e}, Loss: {loss.item() * cfg.training.gradient_accumulation_steps:.4f}, Avg Loss (epoch): {avg_loss_so_far:.4f}")

                if wandb_run:
                    wandb.log({
                        "epoch": epoch + 1,
                        "batch_idx": batch_idx + 1,
                        "train_loss_step": loss.item() * cfg.training.gradient_accumulation_steps,
                        "learning_rate": current_lr,
                        "global_step": global_step
                    })
                
                # Step-based checkpointing
                if cfg.training.save_checkpoint_interval_steps > 0 and global_step % cfg.training.save_checkpoint_interval_steps == 0:
                    log.info(f"[DEBUG] Checkpoint condition: global_step ({global_step}) % interval ({cfg.training.save_checkpoint_interval_steps}) == 0")
                    log.info(f"Saving step-based checkpoint at global_step {global_step}")
                    chkpt_metrics = { "train_loss": avg_loss_so_far, "learning_rate": current_lr }
                    checkpoint_manager.save_checkpoint(
                        model, optimizer, lr_scheduler, 
                        epoch=current_epoch_for_save, step=global_step, metrics=chkpt_metrics, 
                        is_best=False 
                    )

                # Check for max_global_steps limit
                if max_global_steps > 0 and global_step >= max_global_steps:
                    log.info(f"Reached max_global_steps ({max_global_steps}). Stopping training.")
                    training_should_continue = False
                    break # Break inner (batch) loop
            
        # End of epoch logging
        avg_epoch_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        log.info(f"--- End of Epoch {epoch+1}/{cfg.training.epochs} ---")
        log.info(f"Average Training Loss for Epoch: {avg_epoch_loss:.4f}")
    if wandb_run:
            wandb.log({"epoch_train_loss": avg_epoch_loss, "epoch": epoch + 1, "global_step": global_step})
        
        # --- Perform Validation ---
        if cfg.training.perform_validation and val_dataloader and (epoch + 1) % cfg.training.validation_interval_epochs == 0:
            log.info(f"--- Performing validation for Epoch {epoch+1} ---")
            val_loss, val_perplexity = evaluate_model(model, val_dataloader, device, tokenizer.pad_token_id, use_amp, amp_dtype)
            log.info(f"Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.2f}")
            if wandb_run:
                wandb.log({
                    "epoch_val_loss": val_loss,
                    "epoch_val_perplexity": val_perplexity,
                    "epoch": epoch + 1,
                    "global_step": global_step
                })

            # Check for improvement for best model checkpointing
            is_best_checkpoint = False
            if cfg.training.save_best_model_mode == "min":
                if val_loss < best_val_metric - cfg.training.early_stopping_min_delta:
                    best_val_metric = val_loss
                    is_best_checkpoint = True
                    epochs_without_improvement = 0
                    log.info(f"New best validation loss: {best_val_metric:.4f}")
                else:
                    epochs_without_improvement += 1
            else: # mode == "max"
                # Assuming higher is better for metrics like accuracy (not perplexity/loss)
                # This part would need adjustment if using a max metric.
                # For now, val_loss is min, val_perplexity is min.
                # Let's stick to val_loss for early stopping metric directly.
                pass # Add logic for max mode if other metrics are used

            if is_best_checkpoint:
                checkpoint_manager.save_checkpoint(
                    model, optimizer, lr_scheduler,
                    epoch=current_epoch_for_save, step=global_step, 
                    metrics={"train_loss": avg_epoch_loss, "val_loss": val_loss, "val_perplexity": val_perplexity, "learning_rate": current_lr},
                    is_best=True # This will save as best_model.pt
                )

            # Early stopping logic
            if cfg.training.early_stopping_enabled:
                if epochs_without_improvement >= cfg.training.early_stopping_patience:
                    log.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement on {cfg.training.early_stopping_metric}.")
                    training_should_continue = False
                    # break # Break epoch loop is handled by training_should_continue flag
        
        total_loss = 0 # Reset for next epoch
        if not training_should_continue: # Check flag to break outer epoch loop
            break

    if not training_should_continue : # If max_global_steps was reached or early stopping
        log.info("Training terminated due to max_global_steps or early stopping.")

    log.info("Training finished.")
    
    # Save final checkpoint
    log.info("Saving final checkpoint...")

    # Determine the metrics for the final checkpoint
    # If the training loop ran at least one batch_idx, avg_loss_so_far and current_lr should be defined.
    # Otherwise, they might not be, especially if max_global_steps was <=0 or very small from resume.

    # Simpler: use values from the last processed step if they exist, else default.
    # `current_lr` is from the last optimizer step. `avg_loss_so_far` is from within batch loop.
    # Need to ensure these are available or have defaults if the loop didn't run enough.

    # Safest: Use the last known `avg_loss_so_far` and `current_lr` from the loop if it ran.
    # If the loop was completely skipped (e.g., max_global_steps <= global_step at start of loop),
    # then these might not be set. We need to handle this. 
    # `avg_epoch_loss` is only calculated at the end of a *full* epoch. 

    # Let's use the `loaded_metrics` if resuming and no new steps were taken, 
    # or the latest `avg_loss_so_far` and `current_lr` from the loop.

    # Simplification: If the loop for batches was entered, `total_loss` and `current_lr` exist. 
    # `avg_loss_so_far` would also exist. 
    # If no batches were run in the *final* epoch (e.g. max_steps hit before first batch of a new epoch),
    # then these might refer to the previous epoch or be undefined if it was the very first epoch and no batches ran.

    # Revised approach: try to get last values, otherwise use safe defaults or loaded metrics.
    try:
        # These would be from the last completed step if the loop ran
        final_train_loss = avg_loss_so_far 
        final_lr = current_lr
    except NameError: # If loop didn't run enough for these to be set in current scope
        log.warning("Final loss/lr not available from training loop, using last known from checkpoint or defaults.")
        if 'loaded_metrics' in locals() and loaded_metrics: # from checkpoint loading
            final_train_loss = loaded_metrics.get('train_loss', 0.0)
            final_lr = loaded_metrics.get('learning_rate', 0.0)
        else: # Absolute fallback
            final_train_loss = 0.0
            final_lr = 0.0

    final_metrics = {
        "train_loss": final_train_loss,
        "learning_rate": final_lr
    }

    checkpoint_manager.save_checkpoint(
        model, optimizer, lr_scheduler, 
        epoch=current_epoch_for_save, # Use the reliably set current_epoch_for_save
        step=global_step, 
        metrics=final_metrics, 
        is_best=False # Could be best if it's the only one and val was good.
    )

    if wandb_run:
        wandb.finish()
        log.info("W&B run finished.")

    log.info("Hydra application finished.")

# --- Evaluation Function (New) ---
def evaluate_model(model, val_dataloader, device, pad_token_id, use_amp, amp_dtype):
    model.eval() # Set model to evaluation mode
    total_val_loss = 0
    total_tokens_processed = 0 # For perplexity calculation, excluding padding
    log = get_logger(__name__) # Get logger for this function

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            labels = input_ids # For language modeling, labels are the same as input_ids
            
            if use_amp:
                with amp.autocast(device_type=device.type, dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs[0] # outputs[0] is the loss
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs[0]
            
            total_val_loss += loss.item() * input_ids.size(0) # Multiply by batch size (actual items in batch)
            
            # For perplexity, count non-padding tokens in the loss calculation
            # The model's CrossEntropyLoss should handle ignore_index for pad_token_id internally
            # if model.config.pad_token_id is set and matches tokenizer.pad_token_id.
            # Here, we sum tokens that contributed to the loss.
            # Assumes CrossEntropyLoss reduces mean over non-ignored tokens.
            # A simpler way if loss is already per-token (mean): total_val_loss += loss.item() * num_non_pad_tokens
            # For now, let's assume the loss from model is sum over sequence then mean over batch.
            # So, loss.item() is average loss per sequence in batch.
            # To get perplexity, we need sum of losses / total non-pad tokens.
            
            # Let's refine perplexity calculation. The reported loss is typically mean loss per token.
            # So, total_val_loss will be sum of batch_losses. We need to average it by total batches.
            # No, loss.item() is the sum of losses in the batch divided by batch_size * sequence_length (approx)
            # or just sum of losses / batch_size if reduction is mean over batch.
            # If model.forward returns loss that is already a mean over tokens in batch:
            # total_val_loss += loss.item() # then average by len(val_dataloader)
            # For now, sticking to sum(loss_per_batch_item * num_items_in_batch) / total_items
            
            # Simpler: accumulate loss.item() which is already the mean loss for the batch
            # total_val_loss += loss.item() # This is if loss is already mean of batch
            # The current: total_val_loss += loss.item() * input_ids.size(0) assumes loss.item() is mean loss PER SEQUENCE
            # Let's assume HuggingFace model output loss is total loss for batch / batch_size.
            # So, total_val_loss should just sum these mean losses and then average over number of batches.
            # REVERTING accumulation of total_val_loss for now to simple sum of loss.item()
            # total_val_loss += loss.item() # Will average later by len(val_dataloader)

    avg_val_loss = total_val_loss / len(val_dataloader.dataset) if len(val_dataloader.dataset) > 0 else 0
    # avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0 # If loss.item() is mean batch loss
    
    perplexity = 0.0
    if avg_val_loss > 0:
        try:
            perplexity = math.exp(avg_val_loss)
        except OverflowError:
            perplexity = float('inf')
            log.warning(f"Overflow encountered when calculating perplexity from avg_val_loss: {avg_val_loss}. Setting perplexity to infinity.")

    model.train() # Set model back to training mode
    return avg_val_loss, perplexity

if __name__ == "__main__":
    main_app() 