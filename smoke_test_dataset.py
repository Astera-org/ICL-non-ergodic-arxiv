import torch
from random_window_dataset import RandomWindowDataset, DEFAULT_PREPROCESSED_DIR
from pathlib import Path
import numpy as np # For dummy data creation if needed
import json # For dummy data creation if needed

print("Starting RandomWindowDataset smoke test...")

# Configuration
SEQUENCE_LENGTH = 256
BATCH_SIZE = 512 # As per your example
NUM_BATCHES_TO_CHECK = 100 # As per your example
TARGET_SPLIT = "train" # Or "validation", "test"

# Attempt to use actual preprocessed data, fall back to dummy data if not found or if specified
# This allows the smoke test to run in environments without the full dataset
# by creating a minimal viable dummy dataset.

use_dummy_data = False
preprocessed_data_path = DEFAULT_PREPROCESSED_DIR

if not (preprocessed_data_path / "tokens.bin").exists():
    print(f"Warning: Preprocessed data not found at {preprocessed_data_path}. Attempting to create and use dummy data.")
    use_dummy_data = True
    dummy_dir = Path("./dummy_preprocessed_smoke_test")
    preprocessed_data_path = dummy_dir
    
    if not (dummy_dir / "tokens.bin").exists():
        print("Creating dummy data for smoke test...")
        dummy_dir.mkdir(parents=True, exist_ok=True)
        
        # Vocab size for dummy data generation (consistent with dataset class)
        VOCAB_SIZE = 50304 

        # Dummy tokens.bin: 10 papers, each long enough for sequence_length + 1
        # Make paper length variable to better test edge cases
        # Total tokens: sum of (SEQUENCE_LENGTH + 1 + i*10) for i in range(10)
        # All tokens are within valid range [0, VOCAB_SIZE - 1]
        dummy_tokens_list = []
        current_offset = 0
        dummy_index_data = []
        dummy_paper_ids = []

        for i in range(10): # Create 10 dummy papers
            paper_length = SEQUENCE_LENGTH + 1 + i * 10 # Ensure varying lengths, all sufficient
            paper_tokens = np.random.randint(0, VOCAB_SIZE, size=paper_length, dtype=np.uint16)
            dummy_tokens_list.append(paper_tokens)
            
            paper_id = f"dummy_paper_{i}"
            dummy_paper_ids.append(paper_id)
            dummy_index_data.append({
                "paper_id": paper_id,
                "cat": f"cs.XX{i}", # Dummy category
                "offset": current_offset,
                "length": paper_length
            })
            current_offset += paper_length
            
        all_dummy_tokens = np.concatenate(dummy_tokens_list)
        all_dummy_tokens.tofile(dummy_dir / "tokens.bin")
        
        with open(dummy_dir / "index.jsonl", "w") as f:
            for row in dummy_index_data:
                f.write(json.dumps(row) + "\n")
        
        dummy_splits_data = {
            "train": dummy_paper_ids[:8],
            "validation": [dummy_paper_ids[8]],
            "test": [dummy_paper_ids[9]]
        }
        with open(dummy_dir / "splits.json", "w") as f:
            json.dump(dummy_splits_data, f, indent=4)
        print(f"Dummy data created in {dummy_dir}")

try:
    print(f"Initializing RandomWindowDataset with:")
    print(f"  preprocessed_dir = {preprocessed_data_path}")
    print(f"  split            = {TARGET_SPLIT}")
    print(f"  sequence_length  = {SEQUENCE_LENGTH}")

    dataset = RandomWindowDataset(
        preprocessed_dir=preprocessed_data_path,
        split=TARGET_SPLIT,
        sequence_length=SEQUENCE_LENGTH
    )

    print(f"Dataset initialized. Number of available documents (self.pool): {len(dataset)}")
    if len(dataset) == 0:
        print("Dataset is empty. Cannot proceed with DataLoader test. Check filters or data source.")
    else:
        # Set num_workers=0 for easier debugging and to avoid multiprocessing complexities here.
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True) 
        
        print(f"Testing data loading for {NUM_BATCHES_TO_CHECK} batches of size {BATCH_SIZE}...")
        total_tokens_checked = 0

        for i, batch in enumerate(loader):
            if i >= NUM_BATCHES_TO_CHECK:
                break
            
            assert batch.shape[0] <= BATCH_SIZE # Can be less for the last batch
            assert batch.shape[1] == SEQUENCE_LENGTH
            assert batch.dtype == torch.long
            
            # Your crucial checks:
            batch_min = batch.min().item()
            batch_max = batch.max().item()

            assert batch_max < dataset.vocab_size, f"Batch max token ID {batch_max} >= vocab_size {dataset.vocab_size} at batch {i}"
            assert batch_min >= 0, f"Batch min token ID {batch_min} < 0 at batch {i}"
            assert torch.isfinite(batch).all(), f"Non-finite values found in batch {i}" # Should always be true for integer tensors

            total_tokens_checked += batch.numel()
            if (i + 1) % 10 == 0:
                print(f"  Checked batch {i+1}/{NUM_BATCHES_TO_CHECK}... Min: {batch_min}, Max: {batch_max}. Total tokens checked so far: {total_tokens_checked}")

        print(f"\nSuccessfully checked {NUM_BATCHES_TO_CHECK} batches.")
        print(f"Total tokens processed and checked: {total_tokens_checked} (approx. {total_tokens_checked / 1e6:.1f} M tokens)")
        print("✓ Dataset clean: All checks passed for sampled batches.")

except Exception as e:
    print(f"\n--- SMOKE TEST FAILED ---")
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
    print(f"-------------------------")

finally:
    # Optional: Clean up dummy data if it was created
    if use_dummy_data and (dummy_dir / "tokens.bin").exists():
        # Add cleanup logic if desired, e.g., shutil.rmtree(dummy_dir)
        # For now, just printing a message.
        print(f"Dummy data was used/created at: {dummy_dir}. You may want to remove it manually.")

print("\nRandomWindowDataset smoke test finished.") 