# Placeholder for the Python code provided in the previous assistant turn
# The user will save this code locally as debug_embeddings_locally.py

import torch
import json
from pathlib import Path
import sys
import os # Added os for path joining if needed, though Path is better
import logging # For consistency with train.py logging

# --- Setup ---
# For local run, __file__ should work correctly if script is in repo root
PROJECT_ROOT = Path(__file__).resolve().parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("embedding_debug_local") # Changed logger name slightly for clarity

# --- Configuration from problematic_run_args.json ---
args_path = PROJECT_ROOT / "problematic_run_args.json"
batch_path = PROJECT_ROOT / "problematic_batch.pt"

if not args_path.exists() or not batch_path.exists():
    logger.error(f"Ensure 'problematic_run_args.json' ({args_path.exists()}) and " \
                 f"'problematic_batch.pt' ({batch_path.exists()}) are in {PROJECT_ROOT}")
    # It's better to raise an error if files are missing for a script
    raise FileNotFoundError(f"Missing debug files in {PROJECT_ROOT}. Please ensure they are downloaded to the repository root.")

with open(args_path, 'r') as f:
    run_args_dict = json.load(f)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
run_args = AttrDict(run_args_dict)

logger.info(f"Run arguments loaded. Model: {run_args.model_name_or_path}, Seed: {run_args.seed}")

# --- Device and Seed ---
# For local CPU run, this will correctly select 'cpu'
original_device_setting = run_args.get("device", "cuda" if torch.cuda.is_available() else "cpu") # from train.py logic
# Override for local test to ensure CPU if user wants to force it, otherwise auto-detects.
# For this specific request, we want to test on CPU explicitly if not already.
device = torch.device("cpu") 
logger.info(f"Using device: {device} (Original run device was likely: {original_device_setting})")

torch.manual_seed(run_args.seed)
# No need for cuda.manual_seed_all if running on CPU

# --- Load Problematic Batch ---
logger.info(f"Loading problematic batch from: {batch_path}")
input_ids = torch.load(batch_path, map_location='cpu').to(device) # Explicitly load to CPU, then move (though to(device) where device is cpu is fine)
logger.info(f"Problematic batch input_ids loaded. Shape: {input_ids.shape}, Device: {input_ids.device}")

# --- Check Token ID Range and Values ---
try:
    tokenizer_for_vocab_check = AutoTokenizer.from_pretrained(run_args.model_name_or_path)
    VOCAB_SIZE = tokenizer_for_vocab_check.vocab_size
    logger.info(f"Tokenizer Vocab Size: {VOCAB_SIZE}")
    
    min_token_id = input_ids.min().item()
    max_token_id = input_ids.max().item()
    logger.info(f"Problematic Batch - Min Token ID: {min_token_id}, Max Token ID: {max_token_id}")

    if (input_ids < 0).any().item() or (input_ids >= VOCAB_SIZE).any().item():
        logger.error("PROBLEM: Problematic batch CONTAINS TOKEN IDs OUTSIDE VALID RANGE [0, vocab_size-1]!")
        out_of_range_mask = (input_ids < 0) | (input_ids >= VOCAB_SIZE)
        logger.error(f"Offending tokens: {input_ids[out_of_range_mask]}")
    else:
        logger.info("Problematic batch token IDs are all within the valid vocabulary range.")
    
    logger.info(f"Sample of token IDs from the first sequence of the batch: {input_ids[0, :30].tolist()}")

except Exception as e:
    logger.error(f"Error during token ID check: {e}")


# --- Load Model and Inspect Embedding Layer ---
logger.info(f"\nLoading model: {run_args.model_name_or_path} for embedding layer inspection.")
config = AutoConfig.from_pretrained(run_args.model_name_or_path)
model = AutoModelForCausalLM.from_config(config) # Initialize with random weights
model.to(device) # Move model to CPU
model.eval()
logger.info("Model loaded and moved to CPU.")

embedding_layer = model.gpt_neox.embed_in
logger.info(f"Embedding layer type: {type(embedding_layer)}")

# 1. Check for NaNs/Infs in the embedding layer's weights *before* lookup
logger.info("\n--- Checking Embedding Layer Weights --- ")
embedding_weights = embedding_layer.weight
logger.info(f"Embedding weights shape: {embedding_weights.shape}")
is_weights_finite = torch.isfinite(embedding_weights).all().item()
logger.info(f"Embedding weights all finite: {is_weights_finite}")
if not is_weights_finite:
    logger.error("PROBLEM: Embedding layer weights contain NaN/Inf BEFORE lookup!")
    logger.info(f"  Has NaN: {torch.isnan(embedding_weights).any().item()}")
    logger.info(f"  Has Inf: {torch.isinf(embedding_weights).any().item()}")
else:
    logger.info(f"  Weights Min: {embedding_weights.min().item()}")
    logger.info(f"  Weights Max: {embedding_weights.max().item()}")
    logger.info(f"  Weights Mean: {embedding_weights.mean().item()}")

# 2. Perform manual embedding lookup for the problematic batch
logger.info("\n--- Performing Manual Embedding Lookup --- ")
try:
    with torch.no_grad():
        embedded_output = embedding_layer(input_ids)
    
    logger.info(f"Manual embedding output shape: {embedded_output.shape}")
    is_embedded_output_finite = torch.isfinite(embedded_output).all().item()
    logger.info(f"Manual embedding output all finite: {is_embedded_output_finite}")

    if not is_embedded_output_finite:
        logger.error("PROBLEM: Manual embedding lookup resulted in NaN/Inf on CPU!")
        logger.info(f"  Output Has NaN: {torch.isnan(embedded_output).any().item()}")
        logger.info(f"  Output Has Inf: {torch.isinf(embedded_output).any().item()}")
        
        for b_idx in range(input_ids.shape[0]):
            for s_idx in range(input_ids.shape[1]):
                token_id = input_ids[b_idx, s_idx].item()
                # Create a tensor for the single token ID to pass to the embedding layer
                single_token_id_tensor = input_ids[b_idx, s_idx].unsqueeze(0) # Shape [1]
                if single_token_id_tensor.dim() == 0: # Ensure it's at least 1D
                    single_token_id_tensor = single_token_id_tensor.unsqueeze(0)
                
                single_embedding = embedding_layer(single_token_id_tensor)
                if not torch.isfinite(single_embedding).all():
                    logger.error(f"  NON-FINITE embedding for Token ID: {token_id} at batch_idx={b_idx}, seq_idx={s_idx} on CPU")
                    logger.error(f"    Input token ID tensor: {single_token_id_tensor}")
                    logger.error(f"    Corresponding embedding vector (sample): {single_embedding[0, :10].tolist()}...")
                    raise SystemExit("Found first non-finite single token embedding on CPU.") 
    else:
        logger.info("Manual embedding lookup on CPU produced finite results.")
        logger.info(f"  Output Min: {embedded_output.min().item()}")
        logger.info(f"  Output Max: {embedded_output.max().item()}")
        logger.info(f"  Output Mean: {embedded_output.mean().item()}")

except SystemExit as se:
    logger.info(str(se)) # Print the SystemExit message
except Exception as e:
    logger.error(f"Error during manual embedding lookup on CPU: {e}", exc_info=True)

logger.info("\nLocal CPU embedding debug script finished.") 