import hydra
from omegaconf import DictConfig, OmegaConf
import os

# Ensure the script can find the src module when run from the project root
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_loading import load_micro_decoder_from_config
from src.tokenizer_utils import load_tokenizer_from_config # To get vocab size for param count

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def run_test(cfg: DictConfig) -> None:
    """
    Test script to load the MicroDecoderModel using Hydra configuration.
    """
    print("--- Starting Micro Decoder Load Test ---")

    # For this test, we specifically want to load the micro_decoder model config.
    # We'll override the model part of the loaded global config.
    # Create a new OmegaConf object for the model part to simulate loading micro_decoder.yaml
    # This assumes that micro_decoder.yaml is in configs/model/
    # The main config.yaml already defaults to a model, we need to tell Hydra 
    # to use 'micro_decoder' instead for the 'model' group.
    # This is typically done via command-line: `python script.py model=micro_decoder`
    # For a script, we can compose it or directly load what we need.

    # We will simulate the structure Hydra provides after composition when model=micro_decoder is chosen.
    # The `load_micro_decoder_from_config` expects two arguments:
    # 1. model_cfg: The specific configuration for the model (contents of micro_decoder.yaml)
    # 2. global_cfg: The overall global configuration (which contains tokenizer path)

    # First, load the specific model config for micro_decoder
    # Assuming micro_decoder.yaml is at configs/model/micro_decoder.yaml
    # And this script is in scripts/, so config_path is ../configs
    # Hydra automatically loads the defaults. To test a specific model like micro_decoder,
    # we'd normally run the main app with `model=micro_decoder` override.
    # Here, we ensure the passed `cfg` object already has `cfg.model` pointing to our micro_decoder settings.

    # The cfg object passed to this function by @hydra.main already includes the composed config.
    # To ensure it uses micro_decoder, you would run this script like:
    # `python scripts/test_load_micro_decoder.py model=micro_decoder`
    
    print(f"Hydra composed config (model part):\n{OmegaConf.to_yaml(cfg.model)}")

    if cfg.model.name != "micro_decoder":
        print("Error: This script is intended to be run with 'model=micro_decoder'.")
        print(f"Currently loaded model config: {cfg.model.name}")
        print("Please run as: python scripts/test_load_micro_decoder.py model=micro_decoder")
        # As a fallback for direct execution without override, let's try to load it manually if not correct
        try:
            print("Attempting to manually load micro_decoder config for testing fallback...")
            manual_model_cfg_path = os.path.join(project_root, "configs", "model", "micro_decoder.yaml")
            micro_decoder_specific_cfg = OmegaConf.load(manual_model_cfg_path)
            # Merge this into a minimal global_cfg like structure for the loader
            # The loader expects global_cfg.model.custom_tokenizer_path
            # and model_cfg to be the content of micro_decoder.yaml
            
            # Reconstruct a cfg object that looks like what hydra would pass if model=micro_decoder was used
            # For the loader function, global_cfg.model is used for tokenizer path.
            # model_cfg is the specific model config itself.
            
            # Create a minimal global_cfg structure for the tokenizer path
            # This uses the custom_tokenizer_path from your default_model.yaml as a guess
            # You might need to adjust this if your global config is different
            temp_global_cfg_model_part = OmegaConf.create({
                "custom_tokenizer_path": cfg.model.get("custom_tokenizer_path", "models/custom_tokenizer/custom_bpe_tokenizer.json")
            })
            temp_global_cfg = OmegaConf.create({"model": temp_global_cfg_model_part})

            model_cfg_for_loader = micro_decoder_specific_cfg # This is the content of micro_decoder.yaml
            global_cfg_for_loader = temp_global_cfg
            print(f"Manually prepared model_cfg_for_loader target: {model_cfg_for_loader.get('_target_')}")
            print(f"Manually prepared global_cfg_for_loader tokenizer: {global_cfg_for_loader.model.custom_tokenizer_path}")

        except Exception as e:
            print(f"Could not manually load micro_decoder.yaml for fallback: {e}")
            return
    else:
        # If `model=micro_decoder` was correctly passed via command line:
        model_cfg_for_loader = cfg.model # This is already the resolved micro_decoder config
        global_cfg_for_loader = cfg    # The full global config

    try:
        print(f"\n--- Attempting to load Micro Decoder Model ---")
        print(f"Using model config name: {model_cfg_for_loader.get('name', 'N/A')}")
        print(f"Using _target_: {model_cfg_for_loader.get('_target_', 'N/A')}")

        # The load_micro_decoder_from_config expects the model-specific DictConfig and the global DictConfig
        micro_decoder_model = load_micro_decoder_from_config(
            model_cfg=model_cfg_for_loader, 
            global_cfg=global_cfg_for_loader
        )
        print("\n--- Micro Decoder Model Loaded Successfully! ---")
        print(micro_decoder_model)

        # Calculate and print number of parameters
        num_params = sum(p.numel() for p in micro_decoder_model.parameters())
        num_trainable_params = sum(p.numel() for p in micro_decoder_model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable_params:,}")
        
        # Print model config
        print("\n--- Model Configuration (MicroDecoderConfig) ---")
        print(micro_decoder_model.config)

    except Exception as e:
        print(f"Error loading Micro Decoder Model: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Micro Decoder Load Test Finished ---")

if __name__ == "__main__":
    # This script should be run with hydra override: 
    # `python scripts/test_load_micro_decoder.py model=micro_decoder`
    # The @hydra.main decorator will handle loading the specified config.
    run_test() 