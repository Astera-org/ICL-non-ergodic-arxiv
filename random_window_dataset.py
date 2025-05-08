import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict # Added Dict for type hinting record

# Default path for preprocessed data
DEFAULT_PREPROCESSED_DIR = Path("./preprocessed_arxiv")
# Remove global constant
# EFFECTIVE_WINDOW_SIZE = 101 

class RandomWindowDataset(Dataset):
    def __init__(self,
                 preprocessed_dir: Path = DEFAULT_PREPROCESSED_DIR,
                 split: str = "train",
                 target_categories: Optional[List[str]] = None,
                 sequence_length: int = 256):
        """
        Dataset for loading random token windows from the preprocessed arXiv data.

        Args:
            preprocessed_dir (Path): Directory containing tokens.bin, index.jsonl, splits.json.
            split (str): The data split to use ("train", "validation", or "test").
            target_categories (Optional[List[str]]): If provided, only sample from papers 
                                                     belonging to these categories.
            sequence_length (int): The desired length of token sequences (input part) to return.
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.split = split
        self.target_categories = set(target_categories) if target_categories else None
        self.sequence_length = sequence_length # Store sequence length (this is for the input x, target y will be derived)
        self.vocab_size = 50_304 # Pythia vocab size

        tokens_bin_path = self.preprocessed_dir / "tokens.bin"
        index_jsonl_path = self.preprocessed_dir / "index.jsonl"
        splits_json_path = self.preprocessed_dir / "splits.json"

        if not all([tokens_bin_path.exists(), index_jsonl_path.exists(), splits_json_path.exists()]):
            raise FileNotFoundError(
                f"Required data files not found in {self.preprocessed_dir}. "
                f"Please run the preprocessing script (e.g., fetch_arxiv.py) first."
            )

        print(f"Loading memory-mapped tokens from: {tokens_bin_path}")
        self.mem = np.memmap(tokens_bin_path, dtype=np.uint16, mode="r")
        
        print(f"Loading index from: {index_jsonl_path}")
        with open(index_jsonl_path, "r") as f:
            self.idx = [json.loads(line) for line in f]
            
        print(f"Loading splits map from: {splits_json_path}")
        with open(splits_json_path, "r") as f:
            all_splits_data = json.load(f)

        if self.split not in all_splits_data:
            raise ValueError(f"Split '{self.split}' not found in {splits_json_path}. Available: {list(all_splits_data.keys())}")

        paper_ids_for_split = set(all_splits_data[self.split])
        
        self.pool: List[Dict] = [] # Explicitly type self.pool
        initial_pool_count = 0
        for record in self.idx:
            if record["paper_id"] in paper_ids_for_split:
                initial_pool_count += 1
                if self.target_categories and record["cat"] not in self.target_categories:
                    continue
                
                # Document must be long enough to sample sequence_length + 1 tokens
                # (sequence_length for input, 1 for target)
                if record["length"] >= self.sequence_length + 1:
                    self.pool.append(record)
        
        pool_info_str = f"split '{self.split}' with {len(self.pool)} suitable papers"
        if self.target_categories:
             pool_info_str += f" (filtered from {initial_pool_count} total in split for categories {list(sorted(self.target_categories))})"
        else:
             pool_info_str += f" (from {initial_pool_count} total in split)"

        if not self.pool:
            raise ValueError(
                f"No suitable papers found for {pool_info_str} with minimum length {self.sequence_length + 1}. "
                f"The pool is empty. Check data, preprocessing, and category filters."
            )
        
        print(f"Initialized RandomWindowDataset for {pool_info_str}")

    def __len__(self):
        return len(self.pool) # Number of documents available for sampling

    def _sample_window(self, paper_record: Dict) -> torch.LongTensor:
        num_tokens = paper_record["length"]

        # This check should ideally be redundant due to filtering in __init__, but good as a safeguard.
        if num_tokens < self.sequence_length + 1:
            # This indicates an issue if it's ever reached, as __init__ should filter these.
            raise ValueError(
                f"Paper {paper_record['paper_id']} (length {num_tokens}) is too short for "
                f"sequence_length+1 ({self.sequence_length + 1}). Should have been filtered."
            )

        # Inclusive end for random.randint, so subtract (sequence_length + 1) then add 1 to length of range,
        # or simply subtract (sequence_length + 1) from num_tokens for the max starting index.
        # Max start index in paper for a window of (self.sequence_length + 1) tokens
        max_start_index_in_paper = num_tokens - (self.sequence_length + 1)
        
        if max_start_index_in_paper < 0: # Should be caught by length check above
             raise ValueError(
                f"Calculated max_start_index_in_paper ({max_start_index_in_paper}) is negative for paper "
                f"{paper_record['paper_id']} with {num_tokens} tokens and sequence_length {self.sequence_length}. "
                f"This indicates an issue with the length filtering or calculation."
            )

        start_index_in_paper = random.randint(0, max_start_index_in_paper)
            
        global_start_offset = paper_record["offset"] + start_index_in_paper
        # We need sequence_length + 1 tokens to get input (x) and target (y)
        global_end_offset = global_start_offset + self.sequence_length + 1 
        
        window_tokens = self.mem[global_start_offset:global_end_offset]
        
        # Hard guard: check all token IDs in the sampled window (input + potential target)
        # Ensure they are convertible to long and within vocab limits.
        # np.uint16 can't be negative, so only check upper bound.
        # Convert to a temporary tensor for easier min/max if needed, or check numpy array directly.
        if np.any(window_tokens >= self.vocab_size): # Check before converting to tensor
            # Find problematic tokens for logging
            problematic_indices = np.where(window_tokens >= self.vocab_size)[0]
            problematic_values = window_tokens[problematic_indices]
            print(f"ERROR: Out-of-range token IDs in window from paper {paper_record['paper_id']}. "
                  f"Indices relative to window start: {problematic_indices.tolist()}. "
                  f"Values: {problematic_values.tolist()}. Vocab size: {self.vocab_size}. "
                  f"Window (first 10): {window_tokens[:10].tolist()}")
            raise ValueError("Out-of-range token IDs detected in _sample_window.")
            
        # Return only the input part of the window (length = self.sequence_length)
        # The target will be derived from this (e.g., input_tokens[1:]) or handled by model
        input_tokens = torch.as_tensor(window_tokens[:-1].copy().astype(np.int64), dtype=torch.long)
        return input_tokens

    def __getitem__(self, idx: int): # idx is used to pick from pool for retries, or initial choice
        # The original idx might not be used if we always random.choice.
        # For consistency with typical Dataset behavior where idx means something,
        # let's try to use it, but fall back to random for retries.
        # However, the user's patch implied random sampling anyway, and original code did random.choice.
        # Sticking to random.choice for simplicity and robustness against "bad" idx if __len__ changes.

        for attempt in range(10): # Max 10 retries
            chosen_paper_record = random.choice(self.pool) # Always sample a random paper from the pool
            try:
                return self._sample_window(chosen_paper_record)
            except ValueError as e:
                print(f"Warning: ValueError in _sample_window (attempt {attempt+1}/10) for paper {chosen_paper_record.get('paper_id', 'unknown')}: {e}. Retrying with a new paper.")
                if attempt == 9: # Last attempt failed
                    raise RuntimeError(f"Failed to sample a valid window after 10 attempts. Last error: {e}") from e
        # Should not be reached if loop raises RuntimeError
        raise RuntimeError("Exited __getitem__ retry loop unexpectedly.")

if __name__ == '__main__':
    print("Testing RandomWindowDataset...")
    
    # Create dummy data for testing if preprocessed_arxiv doesn't exist
    dummy_dir = Path("./dummy_preprocessed_arxiv")
    if not (dummy_dir / "tokens.bin").exists():
        print("Creating dummy data for testing...")
        dummy_dir.mkdir(parents=True, exist_ok=True)
        
        # Dummy tokens.bin (10 papers, each 200 tokens, all zeros for simplicity)
        dummy_tokens = np.zeros(10 * 200, dtype=np.uint16)
        for i in range(10):
            dummy_tokens[i*200:(i+1)*200] = i # Put some non-zero data
        dummy_tokens.tofile(dummy_dir / "tokens.bin")
        
        # Dummy index.jsonl
        dummy_index_data = []
        # Ensure dummy categories cover potential test cases
        dummy_cats = ["cs.AI", "math.ST", "cs.CV", "cs.NE", "math.AC", "cs.DS", "cs.PL", "math.GR", "cs.IT", "cs.SY"]
        for i in range(10):
            cat = dummy_cats[i] # Assign a category to each dummy paper
            dummy_index_data.append({
                "paper_id": f"train_{i}", "cat": cat, "offset": i * 200, "length": 200
            })
        with open(dummy_dir / "index.jsonl", "w") as f:
            for row in dummy_index_data:
                f.write(json.dumps(row) + "\n")
        
        # Dummy splits.json
        dummy_splits_data = {
            "train": [f"train_{i}" for i in range(8)], # First 8 for train
            "validation": [f"train_{8}"],                # Next 1 for val
            "test": [f"train_{9}"]                     # Last 1 for test
        }
        with open(dummy_dir / "splits.json", "w") as f:
            json.dump(dummy_splits_data, f, indent=4)
        print("Dummy data created.")
        PREPROCESSED_DATA_PATH = dummy_dir
    else:
        PREPROCESSED_DATA_PATH = DEFAULT_PREPROCESSED_DIR # Use actual data if dummy wasn't made

    try:
        print(f"--- Test Train Split (All Categories) (using {PREPROCESSED_DATA_PATH}) ---")
        train_dataset_all = RandomWindowDataset(preprocessed_dir=PREPROCESSED_DATA_PATH, split="train")
        if len(train_dataset_all) > 0:
            print(f"Number of items in train dataset (all cats): {len(train_dataset_all)}")
            sample_train_all = train_dataset_all[0]
            print(f"Sample from train dataset (all cats): {sample_train_all.shape}, {sample_train_all.dtype}")
            # print(sample_train_all)
            assert sample_train_all.shape == (train_dataset_all.sequence_length,)
            assert sample_train_all.dtype == torch.long
        else:
            print("Train dataset (all cats) is empty or could not be loaded.")

        # --- Test with category filtering ---
        test_categories = ["cs.AI", "math.ST"] # Example categories to filter by
        print(f"\n--- Test Train Split (Categories: {test_categories}) (using {PREPROCESSED_DATA_PATH}) ---")
        try:
            train_dataset_filtered = RandomWindowDataset(
                preprocessed_dir=PREPROCESSED_DATA_PATH, 
                split="train", 
                target_categories=test_categories
            )
            if len(train_dataset_filtered) > 0:
                print(f"Number of items in train dataset (filtered): {len(train_dataset_filtered)}")
                sample_train_filtered = train_dataset_filtered[0]
                print(f"Sample from train dataset (filtered): {sample_train_filtered.shape}, {sample_train_filtered.dtype}")
                # print(sample_train_filtered)
                assert sample_train_filtered.shape == (train_dataset_filtered.sequence_length,)
                assert sample_train_filtered.dtype == torch.long
                # Check if sampled paper's category is correct
                # Need to find the record corresponding to the sampled window, which is tricky without more info
                # Instead, let's check the pool directly
                pool_cats = set(r["cat"] for r in train_dataset_filtered.pool)
                print(f"Categories found in filtered pool: {pool_cats}")
                assert pool_cats.issubset(set(test_categories))
            else:
                print(f"Train dataset (filtered for {test_categories}) is empty.")
        except ValueError as e:
            print(f"Could not load filtered dataset: {e}")
            # This might happen if dummy data doesn't contain these categories

        # --- Test Validation Split (All Categories) ---
        if (PREPROCESSED_DATA_PATH / "splits.json").exists():
             with open(PREPROCESSED_DATA_PATH / "splits.json", "r") as f:
                splits_info = json.load(f)
             if "validation" in splits_info and splits_info["validation"]:
                print(f"\n--- Test Validation Split (All Categories) (using {PREPROCESSED_DATA_PATH}) ---")
                val_dataset = RandomWindowDataset(preprocessed_dir=PREPROCESSED_DATA_PATH, split="validation")
                if len(val_dataset) > 0:
                    print(f"Number of items in val dataset: {len(val_dataset)}")
                    sample_val = val_dataset[0]
                    print(f"Sample from val dataset: {sample_val.shape}, {sample_val.dtype}")
                    assert sample_val.shape == (val_dataset.sequence_length,)
                    assert sample_val.dtype == torch.long
                else:
                    print("Validation dataset is empty or could not be loaded.")
             else:
                print("Validation split not available or empty for testing.")


    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 