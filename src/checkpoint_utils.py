import torch
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import wandb

from .logging_config import get_logger

log = get_logger(__name__)

class CheckpointManager:
    def __init__(self, cfg: DictConfig, hydra_run_dir: str):
        self.cfg = cfg
        self.wandb_enabled = cfg.wandb.get("enabled", False) and wandb.run is not None

        # Determine checkpoint directory
        specified_checkpoint_dir = cfg.training.get("checkpoint_dir", None)
        if specified_checkpoint_dir:
            self.checkpoint_dir = Path(specified_checkpoint_dir)
            # If it's a relative path, make it relative to the original CWD if that's desired,
            # or ensure it's created inside hydra_run_dir if specified relatively.
            # For simplicity, let's assume if specified, it might be absolute or relative to original CWD.
            # For robust behavior, it might be better to enforce it being inside hydra_run_dir or absolute.
        else:
            self.checkpoint_dir = Path(hydra_run_dir) / "checkpoints"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Checkpoint directory set to: {self.checkpoint_dir.resolve()}")

        self.best_metric_value = float('inf') if cfg.training.get("save_best_model_mode", "min") == "min" else float('-inf')
        self.save_best_metric = cfg.training.get("save_best_model_metric", None)
        self.save_best_mode = cfg.training.get("save_best_model_mode", "min")

    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, step: int, metrics: dict, is_best: bool = False, filename_prefix: str = "checkpoint"):
        """Saves a checkpoint of the model, optimizer, scheduler, and training state."""
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'hydra_config_resolved': OmegaConf.to_container(self.cfg, resolve=True), # Save resolved config
            'best_metric_value_at_save': self.best_metric_value # Record what was considered best at this save
        }

        # Filename for step-based or epoch-based checkpoints
        # Example: checkpoint_epoch_001_step_001000.pt
        # For simplicity, we can just use a running step count or a generic name like "latest.pt"
        # and then a specific name for "best_model.pt"

        # Save the current checkpoint as "latest.pt" for easy resume
        latest_filename = "latest.pt"
        latest_filepath = self.checkpoint_dir / latest_filename
        torch.save(state, latest_filepath)
        log.info(f"Checkpoint saved to {latest_filepath.resolve()}")

        if is_best:
            best_filename = "best_model.pt"
            best_filepath = self.checkpoint_dir / best_filename
            torch.save(state, best_filepath)
            log.info(f"Best model checkpoint saved to {best_filepath.resolve()} (Metric: {self.save_best_metric} = {metrics.get(self.save_best_metric, 'N/A')})")
            # Update the tracked best metric value
            if self.save_best_metric and self.save_best_metric in metrics:
                 self.best_metric_value = metrics[self.save_best_metric]

        # W&B Artifact Logging (Disabled as per user request)
        """
        if self.wandb_enabled:
            try:
                artifact_name = f"{wandb.run.name if wandb.run else 'run'}-checkpoint"
                artifact = wandb.Artifact(name=artifact_name, type="model")
                
                # Add both latest and best if they are the same file initially, or specific files
                artifact.add_file(str(latest_filepath.resolve()), name="latest_checkpoint.pt")
                if is_best:
                    # If best_filepath is different or just to be explicit
                    best_filepath_to_log = self.checkpoint_dir / "best_model.pt"
                    if best_filepath_to_log.exists(): # Ensure it was actually saved
                         artifact.add_file(str(best_filepath_to_log.resolve()), name="best_model_checkpoint.pt")
                
                aliases = ["latest"]
                if is_best:
                    aliases.append("best")
                
                # Include epoch, step, and metrics in metadata
                metadata = {
                    "epoch": epoch,
                    "step": step,
                    **{f"metric_{k}": v for k,v in metrics.items()} # Flatten metrics for easy W&B filtering
                }
                if self.save_best_metric:
                    metadata[f"best_monitored_metric"] = self.save_best_metric
                    metadata[f"best_monitored_value_at_save"] = self.best_metric_value

                wandb.log_artifact(artifact, aliases=aliases, metadata=metadata)
                log.info(f"Checkpoint artifact '{artifact_name}' logged to W&B with aliases: {aliases}")
            except Exception as e:
                log.error(f"Failed to log checkpoint artifact to W&B: {e}", exc_info=True)
        """
        pass # End of save_checkpoint method

    def load_checkpoint(self, model, optimizer=None, scheduler=None, filename: str = "latest.pt", device=None, load_from_wandb_artifact: str = None):
        """
        Loads a checkpoint into the model, optimizer, and scheduler.
        Can load from a local file or a W&B artifact.

        Args:
            model: The model instance to load state into.
            optimizer: The optimizer instance to load state into (optional).
            scheduler: The scheduler instance to load state into (optional).
            filename: Name of the local checkpoint file (e.g., "latest.pt", "best_model.pt"). Used if load_from_wandb_artifact is None.
            device: The device to map the loaded checkpoint to (e.g., "cpu", "cuda").
            load_from_wandb_artifact: Name of the W&B artifact to load from (e.g., "project/run-name-checkpoint:latest").
                                      If provided, `filename` is ignored for path resolution but might be used to select file from artifact.
                                      THIS IS CURRENTLY DISABLED AS PER USER REQUEST.

        Returns:
            A tuple (epoch, step, metrics, hydra_cfg) if successful, else None.
            hydra_cfg is the resolved config saved in the checkpoint, useful for restoring exact experiment conditions.
        """
        checkpoint_path = None
        # loaded_from_wandb = False # No longer attempting to load from W&B

        # W&B Artifact loading disabled as per user request.
        """
        if load_from_wandb_artifact and self.wandb_enabled:
            try:
                log.info(f"Attempting to load checkpoint from W&B artifact: {load_from_wandb_artifact} (CURRENTLY DISABLED)")
                # artifact = wandb.use_artifact(load_from_wandb_artifact, type='model')
                # artifact_dir = Path(artifact.download())
                # # Try to find the specific file (e.g., latest_checkpoint.pt or best_model_checkpoint.pt) within the artifact dir
                # # The `filename` arg can give a hint if multiple .pt files are in the artifact
                # # For now, let's assume common names or prioritize based on `filename` arg.
                # potential_files = ["latest_checkpoint.pt", "best_model_checkpoint.pt", "checkpoint.pt"]
                # if filename and Path(filename).name.endswith('.pt'): # e.g. if filename was "best_model.pt"
                #     potential_files.insert(0, Path(filename).name) 
                # 
                # for fname_in_artifact in potential_files:
                #     if (artifact_dir / fname_in_artifact).exists():
                #         checkpoint_path = artifact_dir / fname_in_artifact
                #         break
                # 
                # if not checkpoint_path:
                #     log.error(f"Could not find a suitable .pt file in W&B artifact {load_from_wandb_artifact} at {artifact_dir}")
                #     return None
                # log.info(f"Successfully downloaded W&B artifact to {artifact_dir}. Using checkpoint file: {checkpoint_path}")
                # loaded_from_wandb = True
                log.warning("Loading checkpoints from W&B artifacts is currently disabled.")
                return None # Explicitly do not load if W&B artifact path is given while disabled
            except Exception as e:
                log.error(f"Error attempting to handle W&B artifact path (feature disabled) '{load_from_wandb_artifact}': {e}", exc_info=True)
                return None
        """
        # Determine the full path to the checkpoint file
        potential_path = Path(filename)
        if potential_path.is_absolute():
            filepath = potential_path
            log.info(f"Loading checkpoint from absolute path: {filepath}")
        else:
            # If relative, assume it's in the manager's checkpoint_dir (current run's checkpoint dir usually)
            filepath = self.checkpoint_dir / filename
            log.info(f"Loading checkpoint from relative path (in manager's dir): {filepath}")

        if filepath.exists():
            log.info(f"Loading checkpoint from local file: {filepath}")
            try:
                # Load to CPU first to avoid GPU OOM issues if loading a big model, then move
                # Set weights_only=False as our checkpoint contains non-tensor objects (OmegaConf config)
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                log.info("Model state loaded successfully.")

                if optimizer and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    log.info("Optimizer state loaded successfully.")
                elif optimizer:
                    log.warning("Optimizer state not found in checkpoint or optimizer not provided.")

                if scheduler and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    log.info("Scheduler state loaded successfully.")
                elif scheduler:
                    log.warning("Scheduler state not found in checkpoint or scheduler not provided.")

                start_epoch = checkpoint.get('epoch', 0) + 1 # Resume from the next epoch
                global_step = checkpoint.get('step', 0)
                loaded_metrics = checkpoint.get('metrics', {})
                # Hydrated config also available if needed: checkpoint.get('hydra_config_resolved', {})
                
                log.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}, global_step {global_step}.")
                
                # Move model to the target device after loading state dict
                if device:
                    model.to(device)
                    log.info(f"Model moved to device: {device} after loading checkpoint.")
                
                return start_epoch, global_step, loaded_metrics, checkpoint.get('hydra_config_resolved', {})

            except Exception as e:
                log.error(f"Failed to load checkpoint from {filepath}: {e}", exc_info=True)
                return None
        else:
            log.warning(f"Checkpoint file not found at {filepath}. Cannot load checkpoint.")
            return None

    def _is_better(self, current_metric_value):
        if self.save_best_mode == "min":
            return current_metric_value < self.best_metric_value
        else: # mode == "max"
            return current_metric_value > self.best_metric_value

if __name__ == '__main__':
    # Minimal example for syntax checking, assuming dummy configs and objects
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10,1)
        def forward(self, x):
            return self.linear(x)

    dummy_cfg = OmegaConf.create({
        "wandb": {"enabled": False, "project": "test_proj", "entity": "test_entity"},
        "training": {
            "checkpoint_dir": None, # Test default behavior
            "save_best_model_metric": "val_loss",
            "save_best_model_mode": "min"
        },
        "hydra": { # For hydra_run_dir if not explicitly creating it
            "runtime": {
                "output_dir": "./outputs/runs/test_run"
            }
        }
    })

    # Ensure dummy hydra run dir exists for the test
    Path(dummy_cfg.hydra.runtime.output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_manager = CheckpointManager(cfg=dummy_cfg, hydra_run_dir=dummy_cfg.hydra.runtime.output_dir)
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    log.info("Testing save_checkpoint (placeholder)")
    checkpoint_manager.save_checkpoint(model, optimizer, scheduler, epoch=1, step=100, metrics={"train_loss": 0.5, "val_loss": 0.4})

    log.info("Testing load_checkpoint (placeholder)")
    checkpoint_manager.load_checkpoint(model, optimizer, scheduler)

    log.info("CheckpointManager basic structure test complete.") 