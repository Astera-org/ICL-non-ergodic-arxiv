import sys
from pathlib import Path
import torch # Should be imported before our local modules if there are name conflicts
from transformers import AutoTokenizer
import os # Added

# Add project root to sys.path to allow importing RandomWindowDataset
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from random_window_dataset import RandomWindowDataset, DEFAULT_PREPROCESSED_DIR, EFFECTIVE_WINDOW_SIZE # noqa

# Output directory and file
OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "random_window_dataset_sampling_output.txt"

# Tokenizer (consistent with other scripts)
TOKENIZER_NAME = "EleutherAI/pythia-70m-deduped"

NUM_WINDOWS_TO_SAMPLE = 5
SPLIT_TO_TEST = "train" # or "validation", "test"

def test_dataset_sampling():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Redirect stdout to the file
    original_stdout = sys.stdout
    with open(OUTPUT_FILE, 'w') as f_out:
        sys.stdout = f_out

        print(f"Attempting to test RandomWindowDataset sampling from: {DEFAULT_PREPROCESSED_DIR}")
        print(f"Ensure you have run `python fetch_arxiv.py` to generate the processed data.")

        data_path_to_use = None
        if not (DEFAULT_PREPROCESSED_DIR / "tokens.bin").exists():
            print(f"Error: Processed data not found in {DEFAULT_PREPROCESSED_DIR}.")
            print("Please run `python fetch_arxiv.py` first.")
            # Try to use dummy data if actual data is not found, leveraging RandomWindowDataset's test block logic
            dummy_dir_in_dataset_module = PROJECT_ROOT / "dummy_preprocessed_arxiv" # Path from RandomWindowDataset's perspective
            
            if not (dummy_dir_in_dataset_module / "tokens.bin").exists():
                 print(f"Dummy data not found at {dummy_dir_in_dataset_module} either. Cannot proceed with test.")
                 sys.stdout = original_stdout # Restore stdout
                 print(f"Error: Neither real nor dummy data found. Could not write output.") # Print error to console
                 return
            else:
                print(f"Using dummy data from {dummy_dir_in_dataset_module} for testing.")
                data_path_to_use = dummy_dir_in_dataset_module
        else:
            data_path_to_use = DEFAULT_PREPROCESSED_DIR
            print(f"Using actual processed data from {data_path_to_use}")


        print("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        except Exception as e:
            print(f"Error loading tokenizer {TOKENIZER_NAME}: {e}")
            sys.stdout = original_stdout # Restore stdout
            print(f"Error: Could not load tokenizer {TOKENIZER_NAME}. Could not write output.") # Print error to console
            return

        print(f"\n--- Testing RandomWindowDataset with split: '{SPLIT_TO_TEST}' ---")
        dataset = None # Initialize dataset variable
        try:
            dataset = RandomWindowDataset(preprocessed_dir=data_path_to_use, split=SPLIT_TO_TEST)
        except Exception as e:
            print(f"Error instantiating RandomWindowDataset for split '{SPLIT_TO_TEST}': {e}")
            if data_path_to_use and "No suitable papers found" in str(e) and data_path_to_use.name == "dummy_preprocessed_arxiv":
                print("This might be because the dummy data in random_window_dataset.py needs to be created.")
                print("Try running `python random_window_dataset.py` once to generate its internal dummy data.")
            # No dataset loaded, exit block cleanly
            dataset = None # Ensure dataset is None
        
        if dataset is None:
            sys.stdout = original_stdout # Restore stdout
            print(f"Error: Failed to load dataset for split '{SPLIT_TO_TEST}'. Could not write output.") # Print error to console
            return
            
        if len(dataset) == 0:
            print(f"Dataset for split '{SPLIT_TO_TEST}' is empty. Cannot sample.")
            # Exit block cleanly
        else:
            print(f"Dataset loaded. Number of items (approx): {len(dataset)}")
            print(f"Sampling {NUM_WINDOWS_TO_SAMPLE} windows...")

            for i in range(NUM_WINDOWS_TO_SAMPLE):
                try:
                    token_window_tensor = dataset[i] # Index doesn't really matter due to random.choice
                    
                    print(f"\nSampled Window {i+1}/{NUM_WINDOWS_TO_SAMPLE}:")
                    print(f"  Shape: {token_window_tensor.shape}, Dtype: {token_window_tensor.dtype}")
                    assert token_window_tensor.shape == (EFFECTIVE_WINDOW_SIZE,)
                    assert token_window_tensor.dtype == torch.long
                    
                    token_ids = token_window_tensor.tolist()
                    print(f"  Token IDs (first 20): {token_ids[:20]}...")
                    
                    # Decode tokens
                    decoded_text_snippet = tokenizer.decode(token_ids, skip_special_tokens=True)
                    print(f"  Decoded Text Snippet: \"{decoded_text_snippet}\"")
                    print("-" * 20)

                except IndexError:
                    print(f"Could not get sample {i} from dataset. Dataset might be smaller than expected or __len__ is too large for actual pool size.")
                    break
                except Exception as e:
                    print(f"Error during sampling/decoding window {i+1}: {e}")
                    import traceback
                    traceback.print_exc()

    # Restore original stdout
    sys.stdout = original_stdout
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    # This is important for windows, not strictly for linux/mac if __file__ is used right
    # but good practice if you ever move RandomWindowDataset to a package structure.
    # Ensure PROJECT_ROOT is correctly identified before sys.path manipulation if this script moves.
    # current_file_path = Path(__file__).resolve()
    # PROJECT_ROOT = current_file_path.parent.parent 
    # if str(PROJECT_ROOT) not in sys.path:
    #     sys.path.insert(0, str(PROJECT_ROOT))
    
    # from random_window_dataset import RandomWindowDataset, DEFAULT_PREPROCESSED_DIR, EFFECTIVE_WINDOW_SIZE # noqa
    
    test_dataset_sampling() 