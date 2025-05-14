"""
Script to analyze the tokenized data stored in an HDF5 file.
It calculates statistics like chunk counts per category and verifies data integrity.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import h5py
import numpy as np
from collections import Counter
from pathlib import Path

from .logging_config import setup_logging, get_logger

log = get_logger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def hydra_entry_point(cfg: DictConfig) -> None:
    # --- Determine HDF5 file path ---
    # Prefer custom_hdf5_chunked_output_path if available, else fall back.
    default_hdf5_path_str = "data/tokenized_data_chunked_len100.hdf5" # Original default
    
    # Check if we are using a custom tokenizer based on model config
    using_custom_tokenizer = bool(cfg.model.get('custom_tokenizer_path'))
    
    if using_custom_tokenizer:
        # If custom tokenizer is specified, strongly prefer the custom output path
        hdf5_path_str = cfg.dataset.get("custom_hdf5_chunked_output_path", "data/custom_tokenized_data_chunked_len100.hdf5")
        log.info(f"Custom tokenizer is configured. Analyzing HDF5 file: {hdf5_path_str}")
    else:
        # If not using custom tokenizer, use the standard hdf5 path
        hdf5_path_str = cfg.dataset.get("hdf5_chunked_output_path", default_hdf5_path_str)
        log.info(f"Using standard tokenizer. Analyzing HDF5 file: {hdf5_path_str}")

    hdf5_file_path = Path(hdf5_path_str)
    
    log_file_name = f"analyze_{hdf5_file_path.stem}.log"
    analyze_hdf5_data(cfg, hdf5_file_path=hdf5_file_path, log_file_name=log_file_name)

def analyze_hdf5_data(cfg: DictConfig, hdf5_file_path: Path, log_file_name: str) -> None:
    # Update the logging configuration dynamically for this specific analysis run
    if not OmegaConf.is_dict(cfg.get('logging')):
        OmegaConf.set_struct(cfg, False) # Allow adding new keys if logging doesn't exist
        cfg.logging = OmegaConf.create()
        OmegaConf.set_struct(cfg, True)
    
    OmegaConf.update(cfg, "logging.log_file", log_file_name, merge=True)

    setup_logging(cfg) # Pass the modified cfg object
    log.info("--- Analyzing Tokenized HDF5 Data ---")
    log.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    hdf5_input_path = hdf5_file_path
    token_chunk_size = cfg.dataset.get("token_chunk_size", 100)

    log.info(f"Loading HDF5 file: {hdf5_input_path}")

    try:
        with h5py.File(hdf5_input_path, 'r') as hf:
            # 1. Print General Statistics from attributes
            log.info("\n--- General Statistics (from HDF5 attributes) ---")
            for key, value in hf.attrs.items():
                log.info(f"  {key}: {value}")
            
            # Load necessary datasets
            token_chunks_dset = hf['token_chunks']
            doc_idx_for_chunk_dset = hf['doc_idx_for_chunk']
            category_for_chunk_dset = hf['category_for_chunk']

            total_chunks_in_file = token_chunks_dset.shape[0]
            log.info(f"Total chunks found in dataset '{token_chunks_dset.name}': {total_chunks_in_file}")

            if total_chunks_in_file == 0:
                log.warning("No chunks found in the HDF5 file. Cannot perform further analysis.")
                return

            # 2. Verify Chunk Lengths
            log.info(f"\n--- Verifying Chunk Lengths (expected: {token_chunk_size}) ---")
            actual_chunk_length = token_chunks_dset.shape[1]
            if actual_chunk_length == token_chunk_size:
                log.info(f"All chunks have the expected length of {token_chunk_size}.")
            else:
                log.error(f"Error: Chunks have length {actual_chunk_length}, but expected {token_chunk_size}!")
                # For a more thorough check, one could iterate or sample, but shape check is usually sufficient.

            # 3. Calculate and Display Chunk Counts per Original Category
            log.info("\n--- Chunk Counts per Original Category ---")
            # Decode category labels from bytes to strings
            category_labels_str = [label.decode('utf-8') for label in category_for_chunk_dset[:]]
            
            category_counts = Counter(category_labels_str)
            log.info("Number of 100-token chunks per category:")
            for category, count in category_counts.most_common():
                log.info(f"  {category}: {count} chunks")
            
            # 4. (Optional) Display unique document indices and their counts if needed
            log.info("\n--- Document Index Statistics ---")
            unique_doc_indices, counts_per_doc_idx = np.unique(doc_idx_for_chunk_dset[:], return_counts=True)
            log.info(f"Number of unique original documents represented in chunks: {len(unique_doc_indices)}")
            if len(unique_doc_indices) > 0:
                log.info(f"  Min chunks from a single document: {np.min(counts_per_doc_idx)}")
                log.info(f"  Max chunks from a single document: {np.max(counts_per_doc_idx)}")
                log.info(f"  Avg chunks from a single document: {np.mean(counts_per_doc_idx):.2f}")

            log.info("\n--- Analysis Complete ---")

    except FileNotFoundError:
        log.error(f"HDF5 file not found at: {hdf5_input_path}")
    except Exception as e:
        log.error(f"An error occurred during HDF5 analysis: {e}", exc_info=True)

    log.info(f"Analysis complete. Log file: {log_file_name}")

if __name__ == "__main__":
    hydra_entry_point() 