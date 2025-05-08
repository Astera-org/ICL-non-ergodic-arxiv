import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional # Added for type hinting

# Default path for preprocessed data
DEFAULT_PREPROCESSED_DIR = Path("./preprocessed_arxiv")
WINDOW_SIZE = 100 # As per EXPERIMENT_PLAN.md, section 1.4 (100-token slice)
# Your spec for RandomWindowDataset read path used 101, 
# but EXPERIMENT_PLAN.md consistently mentions 100-token windows.
# Let's stick to 100 for the actual data slice, and it means we need 100 tokens for features + 1 for target.
# So, a slice of 101 from memmap is correct if we want to predict the 101st token based on the first 100.
# The eval protocol mentions "cross entropy on tokens 1...99 (BOS ignored)".
# If BOS is token 0, then we predict tokens 1 to 99. This means we need 100 tokens total (0 to 99).
# Let's assume the 100-token slice is [t_0, t_1, ..., t_99].
# If BOS is added, context is BOS + t_0...t_98, predict t_1...t_99.
# The spec "Context length 128 (BOS + 100 + pad)" suggests 100 tokens from the paper.
# Let's go with a slice of 100 tokens from the paper, and the model will handle BOS.
# The RandomWindowDataset spec mentions `tokens = self.mem[row["offset"] + start : row["offset"] + start + 101]`
# This slice is of length 101. This is good for typical LM training (input_ids = tokens[:-1], labels = tokens[1:])

EFFECTIVE_WINDOW_SIZE = 101 # Slice 101 tokens to get 100 for input and 100 for shifted labels

class RandomWindowDataset(Dataset):
    def __init__(self,
                 preprocessed_dir: Path = DEFAULT_PREPROCESSED_DIR,
                 split: str = "train",
                 target_categories: Optional[List[str]] = None):
        """
        Dataset for loading random 100-token windows from the preprocessed arXiv data.

        Args:
            preprocessed_dir (Path): Directory containing tokens.bin, index.jsonl, splits.json.
            split (str): The data split to use ("train", "validation", or "test").
            target_categories (Optional[List[str]]): If provided, only sample from papers 
                                                     belonging to these categories.
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.split = split
        self.target_categories = set(target_categories) if target_categories else None

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
        
        # Filter index to include only papers in the current split and long enough for a window
        # And further filter by target_categories if provided
        self.pool = []
        initial_pool_count = 0
        for record in self.idx:
            if record["paper_id"] in paper_ids_for_split:
                initial_pool_count += 1 # Count papers in this split before length/category filtering
                # Filter by target categories first
                if self.target_categories and record["cat"] not in self.target_categories:
                    continue # Skip this paper if its category is not targeted
                
                # Then filter by length
                if record["length"] >= EFFECTIVE_WINDOW_SIZE:
                    self.pool.append(record)
                # else:
                #     if not self.target_categories or record["cat"] in self.target_categories:
                #         # Only warn if the paper wasn't already filtered out by category
                #         print(f"Warning: Paper {record['paper_id']} (cat: {record['cat']}) in split {self.split} is too short ({record['length']} tokens) for window size {EFFECTIVE_WINDOW_SIZE}. Skipping.")
        
        pool_info_str = f"split '{self.split}' with {len(self.pool)} suitable papers"
        if self.target_categories:
             pool_info_str += f" (filtered from {initial_pool_count} total in split for categories {list(sorted(self.target_categories))})" # Sort categories for consistent print
        else:
             pool_info_str += f" (from {initial_pool_count} total in split)"

        if not self.pool:
            raise ValueError(
                f"No suitable papers found for {pool_info_str} with minimum length {EFFECTIVE_WINDOW_SIZE}. "
                f"The pool is empty. Check data, preprocessing, and category filters."
            )
        
        print(f"Initialized RandomWindowDataset for {pool_info_str}")

    def __len__(self):
        # New implementation: Fixed samples per epoch
        # Define an epoch as processing 100 batches worth of samples.
        # Batch size is typically passed to the DataLoader later.
        # Here, we implicitly assume a batch size to define epoch length.
        # Let's use 100 batches * 256 samples/batch (from run script) = 25600 samples
        return 25600 

    def __getitem__(self, idx): # idx is not really used due to random sampling
        # Choose a paper uniformly from the pool for the current split (pool is already filtered)
        chosen_paper_record = random.choice(self.pool)
        
        # Choose a random start position for the window
        # Ensure there's enough room for a 101-token slice
        # randint is inclusive for both ends, so max_start_index ensures the slice ends within the paper.
        if chosen_paper_record["length"] <= EFFECTIVE_WINDOW_SIZE:
             # This case should ideally be rare if pool is filtered correctly, but as a safeguard:
            start_index_in_paper = 0
        else:
            max_start_index_in_paper = chosen_paper_record["length"] - EFFECTIVE_WINDOW_SIZE
            start_index_in_paper = random.randint(0, max_start_index_in_paper)
            
        # Calculate the start and end offset in the global memory-mapped array
        global_start_offset = chosen_paper_record["offset"] + start_index_in_paper
        global_end_offset = global_start_offset + EFFECTIVE_WINDOW_SIZE
        
        # Extract tokens
        tokens_slice = self.mem[global_start_offset:global_end_offset]
        
        # Convert to torch tensor of type long (as expected by embedding layers)
        # Ensure it's a copy, as memmap arrays might not be writable / have other restrictions.
        # Cast to a type compatible with torch.as_tensor before converting to torch.long
        return torch.as_tensor(tokens_slice.copy().astype(np.int64), dtype=torch.long)

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
            assert sample_train_all.shape == (EFFECTIVE_WINDOW_SIZE,)
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
                assert sample_train_filtered.shape == (EFFECTIVE_WINDOW_SIZE,)
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
                    assert sample_val.shape == (EFFECTIVE_WINDOW_SIZE,)
                    assert sample_val.dtype == torch.long
                else:
                    print("Validation dataset is empty or could not be loaded.")
             else:
                print("Validation split not available or empty for testing.")


    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 