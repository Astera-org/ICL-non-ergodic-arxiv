"""
Defines the HDF5WindowLoader for loading fixed-size token windows
from a pre-processed HDF5 file, supporting K-component balanced sampling.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
import random

from .logging_config import get_logger # Assuming logging_config.py is in the same directory

log = get_logger(__name__)

class HDF5WindowLoader(Dataset):
    """
    A PyTorch Dataset to load 100-token windows from an HDF5 file.

    The HDF5 file is expected to contain:
    - 'token_chunks':      Dataset of [N_total_chunks, chunk_size]
    - 'category_for_chunk': Dataset of [N_total_chunks], mapping chunk to category name (bytes)
    - 'doc_idx_for_chunk': Dataset of [N_total_chunks], mapping chunk to original doc index
    - Attributes:
        - 'category_names_in_file': List of all unique category names (strings)
        - 'reference_chunk_budget_smallest_top_category': Integer (e.g., 501760)
        - 'token_chunk_size': Integer (e.g., 100)
    """
    def __init__(self, cfg: DictConfig, active_categories: list[str], seed: int = 42):
        """
        Initializes the loader.
        Performance considerations:
        - This Dataset is designed to be wrapped by `torch.utils.data.DataLoader`.
        - `DataLoader` handles batching, shuffling (if enabled), and multi-process data loading (`num_workers`).
        - For optimal performance, configure `DataLoader` with appropriate `batch_size` and `num_workers > 0`.

        Args:
            cfg: Hydra configuration object. Expected to have dataset.hdf5_chunked_output_path.
            active_categories: A list of category names (strings) to be included in this dataset.
            seed: Random seed for any operations that might require it in the future (e.g., shuffling
                  if we were to do it here, but currently master list is unshuffled).
        """
        self.hdf5_path = cfg.dataset.hdf5_chunked_output_path
        self.active_categories = active_categories
        self.k = len(active_categories)
        self.seed = seed
        # self.rng = random.Random(seed) # Not used currently for selection, but good to have

        if self.k == 0:
            raise ValueError("active_categories list cannot be empty.")

        self.experiment_chunk_indices = [] # List of HDF5 master indices for this experimental dataset
        self.chunk_categories = [] # Category name for each chunk in experiment_chunk_indices
        self.chunk_doc_ids = []    # Original doc ID for each chunk in experiment_chunk_indices

        self._prepare_dataset_indices()

    def _prepare_dataset_indices(self):
        log.info(f"Preparing dataset indices for K={self.k} active categories: {self.active_categories}")
        try:
            with h5py.File(self.hdf5_path, 'r') as hf:
                log.info(f"Opened HDF5 file: {self.hdf5_path}")
                
                # Load necessary metadata and datasets
                all_hdf5_categories_bytes = hf['category_for_chunk'][:]
                all_hdf5_doc_ids = hf['doc_idx_for_chunk'][:]
                
                # Ensure all_hdf5_categories are strings for easier comparison
                all_hdf5_categories_str = [cat_bytes.decode('utf-8') for cat_bytes in all_hdf5_categories_bytes]

                # Use .attrs for attributes
                raw_category_names = hf.attrs.get('selected_categories_in_file', [])
                log.debug(f"Raw HDF5 attr 'selected_categories_in_file': {raw_category_names} (type: {type(raw_category_names)})")
                if hasattr(raw_category_names, 'dtype'):
                    log.debug(f"dtype of raw_category_names: {raw_category_names.dtype}")

                category_names_in_file = []
                for i, name_val in enumerate(raw_category_names):
                    log.debug(f"  Raw name_val[{i}]: '{name_val}' (type: {type(name_val)})")
                    decoded_name = name_val.decode('utf-8') if isinstance(name_val, bytes) else str(name_val)
                    category_names_in_file.append(decoded_name)
                    log.debug(f"  Decoded name_val[{i}] to string: '{decoded_name}' (type: {type(decoded_name)})")
                log.debug(f"Processed category_names_in_file: {category_names_in_file} (type: {type(category_names_in_file)})")

                ref_budget_total_chunks = hf.attrs.get('reference_chunk_budget_smallest_top_category')
                if ref_budget_total_chunks is None:
                    raise ValueError("HDF5 attribute 'reference_chunk_budget_smallest_top_category' not found.")
                
                self.token_chunk_size = hf.attrs.get('token_chunk_size', 100)
                log.info(f"Reference total chunk budget (from smallest top category): {ref_budget_total_chunks}")
                log.info(f"Token chunk size from HDF5: {self.token_chunk_size}")

                target_chunks_per_category = ref_budget_total_chunks // self.k
                log.info(f"Target chunks per active category for this K={self.k} experiment: {target_chunks_per_category}")

                temp_indices_to_add = []

                for active_cat_name in self.active_categories:
                    log.debug(f"Checking active_cat_name: '{active_cat_name}' (type: {type(active_cat_name)})")
                    log.debug(f"Against category_names_in_file: {category_names_in_file}")
                    is_present = active_cat_name in category_names_in_file
                    log.debug(f"Is '{active_cat_name}' present in category_names_in_file? {is_present}")

                    if not is_present:
                        log.warning(f"Active category '{active_cat_name}' not found in HDF5 attribute 'selected_categories_in_file' (list: {category_names_in_file}). Skipping.")
                        continue
                    
                    log.info(f"Processing active category: {active_cat_name}")
                    
                    # Find all HDF5 master indices for the current active category
                    indices_for_this_cat = [
                        i for i, cat_str in enumerate(all_hdf5_categories_str) if cat_str == active_cat_name
                    ]
                    
                    log.info(f"Found {len(indices_for_this_cat)} total chunks for category '{active_cat_name}' in HDF5.")

                    if not indices_for_this_cat:
                        log.warning(f"No chunks found for active category '{active_cat_name}'. It will not contribute to the dataset.")
                        continue
                    
                    if len(indices_for_this_cat) < target_chunks_per_category:
                        log.warning(
                            f"Category '{active_cat_name}' has only {len(indices_for_this_cat)} chunks, "
                            f"less than the target {target_chunks_per_category}. Using all available."
                        )
                        selected_indices_for_cat = indices_for_this_cat # Take all available
                    else:
                        # Select the *first* N chunks as per user request
                        selected_indices_for_cat = indices_for_this_cat[:target_chunks_per_category]
                    
                    temp_indices_to_add.extend(selected_indices_for_cat)
                    log.info(f"Selected {len(selected_indices_for_cat)} chunks for category '{active_cat_name}'.")

                self.experiment_chunk_indices = temp_indices_to_add
                
                # Now populate self.chunk_categories and self.chunk_doc_ids based on the selected indices
                # This avoids re-reading HDF5 for these in __getitem__ if we pre-fetch them
                # For very large K*target_chunks_per_category, this might be memory intensive.
                # Alternative is to look them up in __getitem__. Let's do that for now to save memory.

                log.info(f"Total chunks in this experiment dataset: {len(self.experiment_chunk_indices)}")
                if len(self.experiment_chunk_indices) != self.k * target_chunks_per_category:
                    log.warning(
                        f"Final dataset size {len(self.experiment_chunk_indices)} does not match "
                        f"K * target_chunks_per_category ({self.k * target_chunks_per_category}) "
                        "due to categories having fewer chunks than targeted."
                    )

        except FileNotFoundError:
            log.error(f"HDF5 file not found at {self.hdf5_path}")
            raise
        except Exception as e:
            log.error(f"Error preparing dataset indices from HDF5: {e}", exc_info=True)
            raise

    def __len__(self):
        return len(self.experiment_chunk_indices)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.experiment_chunk_indices)):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.experiment_chunk_indices)}")

        hdf5_master_idx = self.experiment_chunk_indices[idx]

        try:
            # Performance note for __getitem__:
            # The HDF5 file is opened and closed for each item. This is a standard and safe pattern
            # for PyTorch Datasets when used with `DataLoader(num_workers > 0)` as it avoids issues
            # with sharing h5py file handles across multiple processes.
            # The underlying HDF5 library and OS file caching help mitigate performance impact.
            # For `num_workers = 0` (main process only), one could consider opening the file once
            # in __init__ and keeping the handle, but this adds complexity if num_workers > 0 is ever used.
            with h5py.File(self.hdf5_path, 'r') as hf:
                token_chunk = hf['token_chunks'][hdf5_master_idx, :]
                category_name_bytes = hf['category_for_chunk'][hdf5_master_idx]
                original_doc_id = hf['doc_idx_for_chunk'][hdf5_master_idx]
            
            category_name_str = category_name_bytes.decode('utf-8')
            
            return {
                'input_ids': torch.tensor(token_chunk, dtype=torch.long),
                'category_name': category_name_str,
                'original_doc_id': int(original_doc_id)
            }
        except Exception as e:
            log.error(f"Error loading item at index {idx} (HDF5 master index {hdf5_master_idx}): {e}", exc_info=True)
            # Return a placeholder or raise a specific error if appropriate
            # For now, re-raise to make it visible
            raise

# --- Example Usage ---
if __name__ == '__main__':
    from omegaconf import OmegaConf
    import collections

    # Create a dummy Hydra config for testing
    dummy_cfg = OmegaConf.create({
        'dataset': {
            'hdf5_chunked_output_path': 'data/tokenized_data_chunked_len100.hdf5', 
        },
        'log_level': 'INFO' 
    })

    from .logging_config import setup_logging
    setup_logging(dummy_cfg.log_level)

    log.info("--- Comprehensive Testing for HDF5WindowLoader ---")

    # Define active categories for test instances
    # Ensure these categories exist in your HDF5 file's 'category_names_in_file' attribute
    # and have sufficient chunks for the tests.
    # From our HDF5, cs.PL is smallest of top 5 with 501760 chunks.
    # cs.DS has >700k, math.ST >600k, cs.IT >500k
    test_active_categories_k3 = ['cs.DS', 'math.ST', 'cs.IT'] 
    test_active_categories_k1 = ['cs.PL']
    REFERENCE_CHUNK_BUDGET_SMALLEST_TOP_CATEGORY = 501760 # Expected from HDF5 attr

    try:
        # --- Test K=3 Loader ---
        log.info(f"\n--- Initializing and Testing loader with K={len(test_active_categories_k3)} categories: {test_active_categories_k3} ---")
        loader_k3 = HDF5WindowLoader(cfg=dummy_cfg, active_categories=test_active_categories_k3)
        
        expected_len_k3 = (REFERENCE_CHUNK_BUDGET_SMALLEST_TOP_CATEGORY // 3) * 3
        assert len(loader_k3) == expected_len_k3, f"K=3 Loader length mismatch. Expected {expected_len_k3}, got {len(loader_k3)}"
        log.info(f"Loader K=3 initialized. Total items: {len(loader_k3)} (Matches expected)")

        if len(loader_k3) > 0:
            log.info("Fetching and validating first 10 items from K=3 loader (deterministic order):")
            fetched_categories_k3 = collections.defaultdict(int)
            for i in range(min(10, len(loader_k3))):
                item = loader_k3[i]
                assert item['input_ids'].shape == torch.Size([loader_k3.token_chunk_size]), f"Item {i} shape mismatch."
                assert isinstance(item['category_name'], str), f"Item {i} category is not a string."
                assert item['category_name'] in test_active_categories_k3, f"Item {i} category '{item['category_name']}' not in active list."
                fetched_categories_k3[item['category_name']] += 1
                # log.debug(f"  Item {i}: cat: {item['category_name']}, doc_id: {item['original_doc_id']}, tokens: {item['input_ids'][:5].tolist()}")
            log.info(f"Categories distribution in first 10 samples for K=3: {dict(fetched_categories_k3)}")

            # Verify deterministic fetching
            item_0_run1 = loader_k3[0]['input_ids'][:5].tolist()
            item_0_run2 = loader_k3[0]['input_ids'][:5].tolist()
            assert item_0_run1 == item_0_run2, "Deterministic fetch for item 0 failed for K=3 loader."
            log.info("Deterministic fetch for item 0 (K=3) confirmed.")

        # --- Test K=1 Loader ---
        log.info(f"\n--- Initializing and Testing loader with K={len(test_active_categories_k1)} category: {test_active_categories_k1} ---")
        loader_k1 = HDF5WindowLoader(cfg=dummy_cfg, active_categories=test_active_categories_k1)
        
        expected_len_k1 = (REFERENCE_CHUNK_BUDGET_SMALLEST_TOP_CATEGORY // 1) * 1
        assert len(loader_k1) == expected_len_k1, f"K=1 Loader length mismatch. Expected {expected_len_k1}, got {len(loader_k1)}"
        log.info(f"Loader K=1 initialized. Total items: {len(loader_k1)} (Matches expected)")

        if len(loader_k1) > 0:
            log.info("Fetching and validating first 5 items from K=1 loader:")
            for i in range(min(5, len(loader_k1))):
                item = loader_k1[i]
                assert item['input_ids'].shape == torch.Size([loader_k1.token_chunk_size]), f"Item {i} shape mismatch (K=1)."
                assert item['category_name'] == test_active_categories_k1[0], f"Item {i} category mismatch (K=1)."
            log.info(f"First 5 items for K=1 loader validated (category: {test_active_categories_k1[0]}).")

        # --- Test PyTorch DataLoader Integration & Shuffling (Basic Check) ---
        from torch.utils.data import DataLoader
        if len(loader_k3) > 20: # Ensure enough items for a few batches
            log.info("\n--- Testing with PyTorch DataLoader (K=3, shuffle=True) ---")
            # Reduced batch size and number of batches for quicker test
            dl_k3_shuffle = DataLoader(loader_k3, batch_size=4, shuffle=True, num_workers=0)
            
            log.info(f"DataLoader K=3 (shuffled) created. Approx num batches: {len(dl_k3_shuffle)}")
            
            first_batch_epoch1_items = []
            for i, batch in enumerate(dl_k3_shuffle):
                if i == 0:
                    first_batch_epoch1_items = [(cat, doc_id.item()) for cat, doc_id in zip(batch['category_name'], batch['original_doc_id'])]
                    log.info(f"  Epoch 1, Batch 1 categories & doc_ids: {first_batch_epoch1_items}")
                if i >= 1: # Just check a couple of batches
                    break
            
            first_batch_epoch2_items = []
            for i, batch in enumerate(dl_k3_shuffle): # New iterator for "epoch 2"
                if i == 0:
                    first_batch_epoch2_items = [(cat, doc_id.item()) for cat, doc_id in zip(batch['category_name'], batch['original_doc_id'])]
                    log.info(f"  Epoch 2, Batch 1 categories & doc_ids: {first_batch_epoch2_items}")
                if i >= 1:
                    break
            
            if first_batch_epoch1_items and first_batch_epoch2_items and first_batch_epoch1_items != first_batch_epoch2_items:
                log.info("DataLoader shuffle=True seems to be working (first batches of two epochs differ).")
            elif first_batch_epoch1_items and first_batch_epoch2_items:
                log.warning("First batches of two epochs were identical with shuffle=True. This is statistically unlikely but possible for small datasets/batches.")
            else:
                log.warning("Could not properly verify DataLoader shuffling.")
            log.info("DataLoader test section complete.")

        # --- Test Edge Case: Index out of bounds ---
        log.info("\n--- Testing Edge Case: Index out of bounds ---")
        try:
            if len(loader_k3) > 0:
                _ = loader_k3[len(loader_k3)] # Attempt to access out of bounds
                log.error("IndexError was not raised for out-of-bounds access!")
                assert False, "IndexError not raised."
            else:
                log.info("Skipping out-of-bounds test as loader_k3 is empty.")
        except IndexError:
            log.info("Successfully caught IndexError for out-of-bounds access.")
        except Exception as e:
            log.error(f"Unexpected error during out-of-bounds test: {e}", exc_info=True)
            assert False, f"Unexpected error in OOB test: {e}"
            
        # --- Test Edge Case: K=0 (should be caught by __init__) ---
        log.info("\n--- Testing Edge Case: K=0 (empty active_categories) ---")
        try:
            _ = HDF5WindowLoader(cfg=dummy_cfg, active_categories=[])
            log.error("ValueError was not raised for K=0 active_categories!")
            assert False, "ValueError not raised for K=0."
        except ValueError as ve:
            log.info(f"Successfully caught ValueError for K=0: {ve}")
        except Exception as e:
            log.error(f"Unexpected error during K=0 test: {e}", exc_info=True)
            assert False, f"Unexpected error in K=0 test: {e}"

    except FileNotFoundError:
        log.error(
            f"HDF5 file '{dummy_cfg.dataset.hdf5_chunked_output_path}' not found. "
            "Ensure the batch tokenization script (src/batch_tokenize_dataset.py) has run successfully "
            "and the file is in the correct location."
        )
    except AssertionError as ae:
        log.error(f"TEST FAILED: Assertion Error - {ae}", exc_info=True)
    except Exception as e:
        log.error(f"An error occurred during HDF5WindowLoader testing: {e}", exc_info=True)

    log.info("--- HDF5WindowLoader Comprehensive Test Complete ---") 