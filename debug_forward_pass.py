import torch
import argparse
import json
from pathlib import Path
import sys
import logging

# Ensure the project root is in sys.path if this script is in the root
# and random_window_dataset or other modules are directly importable.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Hook function to print tensor stats
def forward_hook_fn(module, input_tensors, output_tensors):
    module_name = module.__class__.__name__
    
    # For some modules, output_tensors might be a tuple
    if isinstance(output_tensors, tuple):
        # Often, the first element is the main tensor (e.g., hidden_states)
        # This might need adjustment based on specific layer outputs
        tensor_to_check = output_tensors[0] 
    else:
        tensor_to_check = output_tensors

    if not isinstance(tensor_to_check, torch.Tensor):
        logger.info(f"Hook on {module_name}: Output is not a tensor (type: {type(tensor_to_check)}). Skipping stats.")
        return

    is_finite = torch.isfinite(tensor_to_check).all().item()
    has_nan = torch.isnan(tensor_to_check).any().item()
    has_inf = torch.isinf(tensor_to_check).any().item()
    
    logger.info(f"--- Hook on: {module_name} ({module._hook_name_tag if hasattr(module, '_hook_name_tag') else ''}) ---")
    logger.info(f"    Output Tensor Shape: {tensor_to_check.shape}")
    logger.info(f"    Output All Finite: {is_finite}")
    logger.info(f"    Output Has NaN: {has_nan}")
    logger.info(f"    Output Has Inf: {has_inf}")
    if not is_finite:
        logger.warning(f"    NON-FINITE TENSOR DETECTED in {module_name}!")
        logger.info(f"    Min: {tensor_to_check.min().item() if not has_nan and not has_inf else 'N/A (due to NaN/Inf)'}")
        logger.info(f"    Max: {tensor_to_check.max().item() if not has_nan and not has_inf else 'N/A (due to NaN/Inf)'}")
        logger.info(f"    Mean: {tensor_to_check.float().mean().item() if not has_nan and not has_inf else 'N/A (due to NaN/Inf)'}") # Mean works better on float
        # Consider raising an error here to stop at the first non-finite tensor
        # raise ValueError(f"Non-finite tensor detected in {module_name}")


def main():
    parser = argparse.ArgumentParser(description="Debug forward pass with a problematic batch.")
    parser.add_argument("--problematic_batch_path", type=str, default="problematic_batch.pt",
                        help="Path to the saved problematic batch tensor (.pt file).")
    parser.add_argument("--problematic_args_path", type=str, default="problematic_run_args.json",
                        help="Path to the saved arguments for the problematic run (.json file).")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., 'cuda', 'cpu'). If None, uses CUDA if available.")
    
    cli_args = parser.parse_args()

    # Load run arguments
    logger.info(f"Loading run arguments from: {cli_args.problematic_args_path}")
    with open(cli_args.problematic_args_path, 'r') as f:
        run_args_dict = json.load(f)
    
    # Convert relevant args (you might need to adjust which ones are essential for model loading)
    # For simplicity, we'll use an object that allows attribute access
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    run_args = AttrDict(run_args_dict)

    logger.info(f"Run arguments loaded. Model: {run_args.model_name_or_path}, Seed: {run_args.seed}")

    # Determine device
    if cli_args.device:
        device = torch.device(cli_args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set seed for reproducibility (especially for model initialization if from_config)
    torch.manual_seed(run_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_args.seed) # if using CUDA

    # Load problematic batch
    logger.info(f"Loading problematic batch from: {cli_args.problematic_batch_path}")
    problematic_input_ids = torch.load(cli_args.problematic_batch_path, map_location='cpu').to(device)
    logger.info(f"Problematic batch loaded. Shape: {problematic_input_ids.shape}, Device: {problematic_input_ids.device}")

    # Load Model
    logger.info(f"Loading model: {run_args.model_name_or_path}")
    config = AutoConfig.from_pretrained(run_args.model_name_or_path)
    # If you had specific config overrides in train.py, apply them here too.
    # e.g., config.use_cache = False (often good for debugging)

    logger.info("Initializing model from config (random initialization)...")
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)
    model.eval() # Set to evaluation mode
    logger.info("Model loaded and set to evaluation mode.")

    # Register hooks
    # We want to hook into key areas: embeddings, attention layers in each block, and final LM head.
    hook_handles = []

    # 1. Embeddings
    # For Pythia, it's usually model.gpt_neox.embed_in
    try:
        model.gpt_neox.embed_in._hook_name_tag = "Embeddings"
        hook_handles.append(model.gpt_neox.embed_in.register_forward_hook(forward_hook_fn))
        logger.info("Registered hook for Embeddings (model.gpt_neox.embed_in)")
    except AttributeError:
        logger.warning("Could not register hook for default Pythia embed_in layer. Structure might differ.")

    # 2. Transformer Blocks (Layers)
    # For Pythia, layers are model.gpt_neox.layers (a ModuleList)
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        for i, layer in enumerate(model.gpt_neox.layers):
            # Hook input to the block (might be less common, but can be useful)
            # layer._hook_name_tag = f"Layer_{i}_Input" # Custom tag
            # hook_handles.append(layer.register_forward_pre_hook(pre_hook_fn)) # If you need inputs

            # Hook output of the entire block
            layer._hook_name_tag = f"Layer_{i}_Output"
            hook_handles.append(layer.register_forward_hook(forward_hook_fn))
            logger.info(f"Registered hook for Layer {i} output")

            # Hook attention mechanism within each layer
            # Pythia: layer.attention
            if hasattr(layer, 'attention'):
                layer.attention._hook_name_tag = f"Layer_{i}_Attention_Output"
                hook_handles.append(layer.attention.register_forward_hook(forward_hook_fn))
                logger.info(f"  Registered hook for Layer {i} Attention output")
                
                # Further, if attention is a PythiaAttention instance, it has q_proj, k_proj, v_proj, dense
                # For simplicity, we'll stick to the output of the main attention block for now.
                # To go deeper:
                # layer.attention.q_proj._hook_name_tag = f"Layer_{i}_Attention_Q_Proj"
                # hook_handles.append(layer.attention.q_proj.register_forward_hook(forward_hook_fn))
                # ... and so on for k_proj, v_proj, dense, and the softmax calculation if accessible.

            # Hook MLP within each layer
            # Pythia: layer.mlp
            if hasattr(layer, 'mlp'):
                layer.mlp._hook_name_tag = f"Layer_{i}_MLP_Output"
                hook_handles.append(layer.mlp.register_forward_hook(forward_hook_fn))
                logger.info(f"  Registered hook for Layer {i} MLP output")
    else:
        logger.warning("Could not find model.gpt_neox.layers. Hooks for transformer blocks will not be set.")

    # 3. Final Layer Norm (if exists before LM head)
    # Pythia: model.gpt_neox.final_layer_norm
    try:
        model.gpt_neox.final_layer_norm._hook_name_tag = "FinalLayerNorm"
        hook_handles.append(model.gpt_neox.final_layer_norm.register_forward_hook(forward_hook_fn))
        logger.info("Registered hook for FinalLayerNorm (model.gpt_neox.final_layer_norm)")
    except AttributeError:
        logger.warning("Could not register hook for default Pythia final_layer_norm layer.")
        
    # 4. LM Head (Output Logits)
    # For Pythia, this is model.embed_out (which is often tied to embed_in weights)
    # If it's just a Linear layer, its output is directly the logits.
    try:
        model.embed_out._hook_name_tag = "LM_Head_Logits (embed_out)"
        hook_handles.append(model.embed_out.register_forward_hook(forward_hook_fn))
        logger.info("Registered hook for LM Head (model.embed_out)")
    except AttributeError:
        logger.warning("Could not register hook for default Pythia embed_out layer.")


    logger.info(f"\n--- Performing forward pass with problematic batch (Shape: {problematic_input_ids.shape}) ---")
    try:
        with torch.no_grad(): # No gradients needed for debugging the forward pass
             # For Pythia, labels are not strictly needed if you only care about logits for debugging
            outputs = model(input_ids=problematic_input_ids) 
            
        logger.info("--- Forward pass completed. --- Final Logits Stats: ---")
        final_logits = outputs.logits
        is_finite = torch.isfinite(final_logits).all().item()
        has_nan = torch.isnan(final_logits).any().item()
        has_inf = torch.isinf(final_logits).any().item()
        logger.info(f"    Final Logits Shape: {final_logits.shape}")
        logger.info(f"    Final Logits All Finite: {is_finite}")
        logger.info(f"    Final Logits Has NaN: {has_nan}")
        logger.info(f"    Final Logits Has Inf: {has_inf}")
        if not is_finite:
            logger.error("    FINAL LOGITS ARE NON-FINITE!")
        else:
            logger.info("    Final logits appear to be finite.")

    except Exception as e:
        logger.error(f"Exception during forward pass: {e}", exc_info=True)
    finally:
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
        logger.info("All hooks removed.")

if __name__ == "__main__":
    main() 