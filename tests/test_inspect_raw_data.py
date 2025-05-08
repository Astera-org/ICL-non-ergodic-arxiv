import sys
import os
from datasets import load_dataset
from pathlib import Path

# Dataset details (consistent with fetch_arxiv.py)
RAW_DATASET_NAME = "ccdv/arxiv-classification"
# Note: If tests/ is run from the root, this should be "./data_cache_raw_arxiv"
# Assuming the test script is run from the `tests/` directory itself, or
# that the cache dir is relative to the script location.
# For robust path handling, it's often better to define paths from project root.
# Let's adjust to be relative to a potential project root execution.
PROJECT_ROOT = Path(__file__).parent.parent # Get project root (one level up from tests/)
RAW_DATA_CACHE_DIR_FROM_ROOT = PROJECT_ROOT / "data_cache_raw_arxiv"

# Output directory and file
OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "inspect_raw_data_output.txt"

NUM_EXAMPLES_TO_SHOW = 3

def inspect_raw_data():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Redirect stdout to the file
    original_stdout = sys.stdout
    with open(OUTPUT_FILE, 'w') as f:
        sys.stdout = f
        
        print(f"Attempting to load raw dataset: {RAW_DATASET_NAME} from cache {RAW_DATA_CACHE_DIR_FROM_ROOT}")
        print(f"Ensure you have run `python fetch_arxiv.py` at least once to populate the cache.")
        
        try:
            # Make sure cache directory exists if we are creating it
            # RAW_DATA_CACHE_DIR_FROM_ROOT.mkdir(parents=True, exist_ok=True) # load_dataset will handle cache creation
            
            dataset_dict = load_dataset(
                RAW_DATASET_NAME, 
                cache_dir=str(RAW_DATA_CACHE_DIR_FROM_ROOT), 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure the dataset is downloaded or accessible.")
            print(f"If you haven't run `fetch_arxiv.py` before, the cache directory might not be initialized by it yet.")
            print(f"The `datasets` library usually caches to `~/.cache/huggingface/datasets` by default if `cache_dir` is problematic.")
            print(f"Trying to load without explicit cache_dir to use default Hugging Face cache...")
            try:
                dataset_dict = load_dataset(RAW_DATASET_NAME, trust_remote_code=True)
            except Exception as e_default:
                print(f"Failed to load with default cache as well: {e_default}")
                sys.stdout = original_stdout # Restore stdout before returning
                return

        print(f"Raw dataset loaded successfully: {dataset_dict}")

        category_names_raw = dataset_dict['train'].features['label'].names
        category_names_cleaned = [name.strip() for name in category_names_raw]

        for split_name in dataset_dict.keys():
            print(f"\n--- Inspecting Split: {split_name} ---")
            dataset_split = dataset_dict[split_name]
            
            if len(dataset_split) == 0:
                print(f"Split {split_name} is empty.")
                continue

            print(f"Total examples in {split_name}: {len(dataset_split)}")
            
            for i in range(min(NUM_EXAMPLES_TO_SHOW, len(dataset_split))):
                example = dataset_split[i]
                text = example['text']
                label_id = example['label']
                category_str = category_names_cleaned[label_id]
                
                print(f"\nExample {i+1}/{min(NUM_EXAMPLES_TO_SHOW, len(dataset_split))} from '{split_name}':")
                print(f"  Label ID: {label_id}")
                print(f"  Category: {category_str}")
                print(f"  Text (first 300 chars): {text[:300]}...")
                print("-" * 20)

    # Restore original stdout
    sys.stdout = original_stdout
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    inspect_raw_data() 