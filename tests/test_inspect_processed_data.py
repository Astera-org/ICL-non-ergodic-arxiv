import json
import random
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import sys
import os

# Define the processed data directory (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_arxiv"

# Output directory and file
OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "inspect_processed_data_output.txt"

# Tokenizer (consistent with fetch_arxiv.py)
TOKENIZER_NAME = "EleutherAI/pythia-70m-deduped"

NUM_EXAMPLES_TO_SHOW_PER_SPLIT = 2
MAX_TOKENS_TO_PRINT = 50
MAX_CHARS_TO_PRINT = 200

def inspect_processed_data():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Redirect stdout to the file
    original_stdout = sys.stdout
    with open(OUTPUT_FILE, 'w') as f_out:
        sys.stdout = f_out

        print(f"Attempting to load processed data from: {PREPROCESSED_DIR}")
        print(f"Ensure you have run `python fetch_arxiv.py` to generate this data.")

        tokens_bin_path = PREPROCESSED_DIR / "tokens.bin"
        index_jsonl_path = PREPROCESSED_DIR / "index.jsonl"
        splits_json_path = PREPROCESSED_DIR / "splits.json"

        if not all([tokens_bin_path.exists(), index_jsonl_path.exists(), splits_json_path.exists()]):
            print(f"Error: Required data files not found in {PREPROCESSED_DIR}.")
            print("Please run `python fetch_arxiv.py` first.")
            sys.stdout = original_stdout # Restore stdout
            print(f"Error: Required data files not found in {PREPROCESSED_DIR}. Could not write output.") # Print error to console
            return

        print("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        except Exception as e:
            print(f"Error loading tokenizer {TOKENIZER_NAME}: {e}")
            sys.stdout = original_stdout # Restore stdout
            print(f"Error: Could not load tokenizer {TOKENIZER_NAME}. Could not write output.") # Print error to console
            return
        
        print("Loading index and splits...")
        with open(index_jsonl_path, "r") as f_idx:
            index_data = [json.loads(line) for line in f_idx]
        
        with open(splits_json_path, "r") as f_splits:
            splits_map = json.load(f_splits)

        print("Loading memory-mapped tokens...")
        mem_tokens = np.memmap(tokens_bin_path, dtype=np.uint16, mode="r")

        for split_name, paper_ids_in_split in splits_map.items():
            print(f"\n--- Inspecting Split: {split_name} ---")
            if not paper_ids_in_split:
                print(f"Split {split_name} is empty.")
                continue

            # Get records from index that are in this split
            split_records = [record for record in index_data if record["paper_id"] in paper_ids_in_split]
            
            if not split_records:
                print(f"No records found in index.jsonl for paper_ids in split '{split_name}'.")
                continue
                
            print(f"Found {len(split_records)} papers in '{split_name}' based on index.")

            for i in range(min(NUM_EXAMPLES_TO_SHOW_PER_SPLIT, len(split_records))):
                # Choose a random paper from this split's records
                chosen_record = random.choice(split_records)
                
                paper_id = chosen_record["paper_id"]
                category = chosen_record["cat"]
                offset = chosen_record["offset"]
                length = chosen_record["length"]

                print(f"\nExample {i+1}/{min(NUM_EXAMPLES_TO_SHOW_PER_SPLIT, len(split_records))} from '{split_name}':")
                print(f"  Paper ID: {paper_id}")
                print(f"  Category: {category}")
                print(f"  Offset: {offset}")
                print(f"  Length: {length} tokens")

                if length == 0:
                    print("  Paper has 0 tokens. Skipping token inspection.")
                    continue

                # Extract tokens
                paper_tokens = mem_tokens[offset : offset + length]
                
                print(f"  First {min(MAX_TOKENS_TO_PRINT, length)} Tokens: {paper_tokens[:MAX_TOKENS_TO_PRINT].tolist()}")
                
                # Decode tokens
                try:
                    decoded_text = tokenizer.decode(paper_tokens, skip_special_tokens=True)
                    print(f"  Decoded Text (first {min(MAX_CHARS_TO_PRINT, len(decoded_text))} chars): {decoded_text[:MAX_CHARS_TO_PRINT]}...")
                except Exception as e:
                    print(f"  Error decoding tokens for {paper_id}: {e}")
                print("-" * 20)

    # Restore original stdout
    sys.stdout = original_stdout
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    inspect_processed_data() 