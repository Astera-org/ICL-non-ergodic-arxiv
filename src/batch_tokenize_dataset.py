"""
Script to load the filtered arXiv dataset, tokenize it in batches,
chunk the token sequences, and save them to an HDF5 file.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import datasets
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time

from src.logging_config import get_logger, setup_logging
from src.utils import set_seed
from src.dataset_loader import ArxivDatasetLoader
from src.tokenizer_utils import load_tokenizer_from_config, chunk_token_ids

log = get_logger(__name__)

# Define project root if not already defined (assuming script is in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTED_CATEGORIES_HARDCODED = ["cs.DS", "math.ST", "math.GR", "cs.IT", "cs.PL"]
TEXT_COLUMN = "text" # Ensure this matches the dataset feature

def main_batch_tokenize(cfg: DictConfig) -> None:
    setup_logging(cfg) # Use Hydra config for logging setup
    set_seed(cfg.seed)
    log.info(f"Starting batch tokenization with config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Determine target split and output path ---
    # Allow overriding target split via cfg, e.g., cfg.dataset.target_split
    target_split_arg = cfg.dataset.get("target_split", "all").lower() # Default to 'all'
    log.info(f"Target data split for processing: {target_split_arg}")

    # Default base output path (used if not fully specified in config)
    default_base_output_path = "data/custom_tokenized_data_chunked_len100"
    
    # Start with the path from config or a sensible default structure
    output_hdf5_path_str_base = cfg.dataset.get("custom_hdf5_chunked_output_path_base", 
                                                cfg.dataset.get("hdf5_chunked_output_path_base", 
                                                                default_base_output_path))

    # Append split name to filename if processing a specific split and not 'all'
    if target_split_arg not in ["all", "all_splits"]: # 'all_splits' as another alias for 'all'
        output_filename = f"{Path(output_hdf5_path_str_base).stem}_{target_split_arg}.hdf5"
    else:
        output_filename = f"{Path(output_hdf5_path_str_base).stem}.hdf5" # Default if 'all'

    # Ensure the directory from the base path is used
    output_hdf5_path = PROJECT_ROOT / Path(output_hdf5_path_str_base).parent / output_filename
    output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Final HDF5 output path: {output_hdf5_path}")

    # --- 1. Load Dataset ---
    log.info("Loading arXiv dataset...")
    loader = ArxivDatasetLoader(config=cfg)
    
    # Load all splits initially to check availability
    available_splits_all = loader.load_all_splits() 

    splits_to_process = {}
    if target_split_arg == "all" or target_split_arg == "all_splits":
        splits_to_process = available_splits_all
        log.info(f"Processing all available splits: {list(splits_to_process.keys())}")
    elif target_split_arg in available_splits_all:
        splits_to_process = {target_split_arg: available_splits_all[target_split_arg]}
        log.info(f"Processing only the '{target_split_arg}' split.")
    else:
        log.error(f"Target split '{target_split_arg}' is not 'all' and not found in available splits: {list(available_splits_all.keys())}. Exiting.")
        return
    
    if not splits_to_process:
        log.error("No dataset splits could be loaded or selected for processing. Exiting.")
        return

    all_docs_for_tokenization = []
    doc_original_indices = [] 
    doc_categories = []       

    log.info(f"Filtering splits by hardcoded top 5 categories: {SELECTED_CATEGORIES_HARDCODED}")
    for split_name, ds_split in splits_to_process.items(): # Iterate over selected splits
        if ds_split is None:
            log.warning(f"Split '{split_name}' is None, skipping.")
            continue
        
        log.info(f"Filtering split: {split_name}...")
        # Get label names specific to this split
        label_names_for_split = loader.get_label_names(ds_split)
        if not label_names_for_split:
            log.warning(f"Could not get label names for split '{split_name}'. Skipping filtering for this split.")
            # Optionally, decide if you want to add all documents from this split or skip
            # For now, let's assume we only process if we can map labels.
            continue

        filtered_ds_split = loader.filter_by_specific_categories(
            ds_split, # Pass the loaded dataset split
            categories_to_keep=SELECTED_CATEGORIES_HARDCODED
        )
        
        if not filtered_ds_split or len(filtered_ds_split) == 0:
            log.info(f"Split '{split_name}' after filtering by categories is empty or None.")
            continue
            
        log.info(f"Processing filtered split: {split_name} with {len(filtered_ds_split)} documents.")
        for i, doc in enumerate(filtered_ds_split):
            all_docs_for_tokenization.append(doc[TEXT_COLUMN])
            
            label_idx = doc['label']
            if 0 <= label_idx < len(label_names_for_split):
                doc_category_name = label_names_for_split[label_idx]
            else:
                # This case should ideally not happen if filtering worked and label names are correct
                log.warning(f"Unexpected label index {label_idx} for split {split_name}, doc {i}. Storing as 'unknown'.")
                doc_category_name = "unknown"
            
            doc_categories.append(doc_category_name) 
            doc_original_indices.append(f"{split_name}_{i}") 

    if not all_docs_for_tokenization:
        log.error("No documents found after filtering and processing all splits. Exiting.")
        return
    log.info(f"Total documents for tokenization: {len(all_docs_for_tokenization)}")

    # --- 2. Load Tokenizer ---
    log.info("Loading tokenizer...")
    tokenizer = load_tokenizer_from_config(cfg.model) # Pass cfg.model here
    if tokenizer is None:
        log.error("Failed to load tokenizer. Exiting.")
        return

    is_hf_tokenizer = hasattr(tokenizer, 'batch_encode_plus')
    log.info(f"Tokenizer type: {'Hugging Face PreTrainedTokenizer' if is_hf_tokenizer else 'Custom tokenizers.Tokenizer'}")

    # --- 3. Tokenize in Batches & Chunk ---
    # Use the new output path from config for custom tokenized data
    # Default to "custom_hdf5_chunked_output_path" if "hdf5_chunked_output_path" (old) is not present
    # and provide a final default if neither is present.
    # The output_hdf5_path is now determined earlier based on target_split_arg

    # output_hdf5_path_str = cfg.dataset.get("custom_hdf5_chunked_output_path", 
    #                                        cfg.dataset.get("hdf5_chunked_output_path", default_custom_output_path))
    
    # If we are using a custom tokenizer, ensure the output path reflects that,
    # unless a specific custom_hdf5_chunked_output_path is already set.
    # if not is_hf_tokenizer and "custom_hdf5_chunked_output_path" not in cfg.dataset:
    #     output_hdf5_path_str = default_custom_output_path
    #     log.info(f"Using custom tokenizer, updated output path to default: {output_hdf5_path_str}")

    # output_hdf5_path = PROJECT_ROOT / output_hdf5_path_str # Path determined earlier
    # output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunk_size = cfg.training.get("token_chunk_size", 100) 
    batch_size = cfg.training.tokenization_batch_size
    
    log.info(f"Tokenizing {len(all_docs_for_tokenization)} documents in batches of {batch_size}, chunk size {chunk_size}...")
    log.info(f"Output will be saved to: {output_hdf5_path}")

    all_chunked_token_ids = []
    all_doc_idx_for_chunk = []
    all_category_for_chunk = [] 

    start_time = time.time()
    for i in tqdm(range(0, len(all_docs_for_tokenization), batch_size), desc="Tokenizing batches"):
        batch_texts = all_docs_for_tokenization[i:i + batch_size]
        batch_categories = doc_categories[i:i+batch_size]

        if is_hf_tokenizer:
            encoded_batch = tokenizer.batch_encode_plus(
                batch_texts, 
                add_special_tokens=False, 
                padding=False, 
                truncation=False 
            )
            batch_token_ids_list = encoded_batch['input_ids']
        else:
            # For custom 'tokenizers.Tokenizer', use encode_batch() which returns a list of Encoding objects
            # Each Encoding object has an 'ids' attribute.
            # The custom tokenizer was trained without explicit add_special_tokens=False,
            # but BPE typically doesn't add CLS/SEP unless specified in post-processing.
            # For safety, one might add `enable_padding=False, enable_truncation=False` to encode_batch
            # if those features were enabled on the tokenizer globally, but usually not needed for raw BPE.
            batch_encodings = tokenizer.encode_batch(batch_texts)
            batch_token_ids_list = [enc.ids for enc in batch_encodings]

        for doc_idx_in_batch, token_ids in enumerate(batch_token_ids_list):
            original_doc_global_idx = i + doc_idx_in_batch 
            doc_category = batch_categories[doc_idx_in_batch]

            chunks = chunk_token_ids(token_ids, chunk_size)
            if chunks:
                all_chunked_token_ids.extend(chunks)
                all_doc_idx_for_chunk.extend([original_doc_global_idx] * len(chunks))
                all_category_for_chunk.extend([doc_category] * len(chunks))
    
    end_time = time.time()
    log.info(f"Tokenization and chunking completed in {end_time - start_time:.2f} seconds.")
    log.info(f"Total chunks created: {len(all_chunked_token_ids)}")

    if not all_chunked_token_ids:
        log.warning("No chunks were created. Check data or chunking parameters.")
        return

    # --- 4. Save to HDF5 ---
    log.info(f"Saving {len(all_chunked_token_ids)} chunks to HDF5 file: {output_hdf5_path}...")
    with h5py.File(output_hdf5_path, 'w') as f:
        f.create_dataset('token_chunks', data=np.array(all_chunked_token_ids, dtype=np.int32))
        f.create_dataset('doc_idx_for_chunk', data=np.array(all_doc_idx_for_chunk, dtype=np.int32))
        
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('category_for_chunk', data=np.array(all_category_for_chunk, dtype=dt))

        f.attrs['total_chunks'] = len(all_chunked_token_ids)
        f.attrs['chunk_size'] = chunk_size
        f.attrs['tokenizer_name_or_path'] = cfg.model.get('tokenizer_name') or cfg.model.get('custom_tokenizer_path', 'unknown')
        
        unique_categories_in_file = sorted(list(set(all_category_for_chunk)))
        f.attrs['selected_categories_in_file'] = unique_categories_in_file
        
        counts_per_category_in_hdf5 = {cat: 0 for cat in unique_categories_in_file}
        for cat_in_chunk in all_category_for_chunk:
            counts_per_category_in_hdf5[cat_in_chunk] += 1
        
        min_chunks_for_top_categories = float('inf')
        # We need to check against the originally targeted SELECTED_CATEGORIES_HARDCODED
        # to ensure the budget is based on the smallest of *those*, if they are present.
        
        present_hardcoded_categories = [cat for cat in SELECTED_CATEGORIES_HARDCODED if cat in counts_per_category_in_hdf5 and counts_per_category_in_hdf5[cat] > 0]

        if present_hardcoded_categories:
            min_chunks_for_top_categories = min(counts_per_category_in_hdf5[cat] for cat in present_hardcoded_categories)
        else:
            log.warning(f"None of the SELECTED_CATEGORIES_HARDCODED ({SELECTED_CATEGORIES_HARDCODED}) were found in the generated HDF5 file's chunks. Cannot accurately set reference_chunk_budget_smallest_top_category based on them.")
            # Fallback: if no hardcoded categories are present, but other categories are,
            # this value might not be meaningful in the original context. Setting to 0.
            min_chunks_for_top_categories = 0
        
        if min_chunks_for_top_categories == float('inf'): # Should only happen if present_hardcoded_categories was empty
            min_chunks_for_top_categories = 0

        f.attrs['reference_chunk_budget_smallest_top_category'] = min_chunks_for_top_categories
        log.info(f"Calculated reference_chunk_budget_smallest_top_category: {min_chunks_for_top_categories} chunks (based on current HDF5 content of hardcoded top 5 categories if present).")

    log.info(f"Successfully saved tokenized data to {output_hdf5_path}")
    log.info(f"Total documents processed: {len(all_docs_for_tokenization)}")
    log.info(f"Total chunks generated: {len(all_chunked_token_ids)}")

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def hydra_entry_point(cfg: DictConfig) -> None:
    main_batch_tokenize(cfg)

if __name__ == "__main__":
    hydra_entry_point()