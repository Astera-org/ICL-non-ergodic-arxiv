import argparse
import json
from pathlib import Path
import numpy as np
import sys
import os

# Add project root to sys.path if this script is not in the root
script_dir = Path(__file__).resolve().parent
project_root = script_dir # Assumes script is in root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to sys.path")


try:
    from random_window_dataset import RandomWindowDataset, DEFAULT_PREPROCESSED_DIR
    # Use ALL_CATEGORIES from train.py if available and needed
    try:
        from train import ALL_CATEGORIES
    except ImportError:
        print("Warning: Could not import ALL_CATEGORIES from train.py. Category validation might be limited.")
        ALL_CATEGORIES = None # Define as None if import fails
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure this script is run from the project root or adjust sys.path.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Diagnose RandomWindowDataset initialization.")
    parser.add_argument("--preprocessed_data_dir", type=Path, default=DEFAULT_PREPROCESSED_DIR, help="Directory with preprocessed ArXiv data.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], help="Data split to diagnose.")
    parser.add_argument("--target_category", type=str, default=None, help="Optional single category to filter for (e.g., 'math.AC').")
    parser.add_argument("--sequence_length", type=int, default=256, help="Sequence length threshold for document suitability.")

    args = parser.parse_args()

    # Ensure the data directory path is resolved correctly
    data_dir = args.preprocessed_data_dir.resolve()


    print(f"--- Diagnosing Dataset ---")
    print(f"Directory: {data_dir}")
    print(f"Split: {args.split}")
    print(f"Target Category: {args.target_category}")
    print(f"Sequence Length (for filtering): {args.sequence_length}")
    min_required_length = args.sequence_length + 1
    print(f"Minimum Required Length (seq_len + 1): {min_required_length}")
    print("-" * 20)

    # Load splits and index manually first for pre-filtering stats
    index_jsonl_path = data_dir / "index.jsonl"
    splits_json_path = data_dir / "splits.json"

    if not index_jsonl_path.exists() or not splits_json_path.exists():
        print(f"ERROR: index.jsonl or splits.json not found in {data_dir}")
        return

    print(f"Loading index from: {index_jsonl_path}")
    try:
        with open(index_jsonl_path, "r") as f:
            full_idx = [json.loads(line) for line in f]
    except Exception as e:
        print(f"ERROR loading or parsing {index_jsonl_path}: {e}")
        return

    print(f"Loading splits map from: {splits_json_path}")
    try:
        with open(splits_json_path, "r") as f:
            all_splits_data = json.load(f)
    except Exception as e:
        print(f"ERROR loading or parsing {splits_json_path}: {e}")
        return


    if args.split not in all_splits_data:
        print(f"ERROR: Split '{args.split}' not found in splits file. Available splits: {list(all_splits_data.keys())}")
        return

    paper_ids_for_split = set(all_splits_data[args.split])
    print(f"Total documents listed in '{args.split}' split: {len(paper_ids_for_split)}")

    # Filter for target category *before* length check
    category_docs_in_split = []
    target_cat_set = {args.target_category} if args.target_category else None

    print("\n--- Pre-filtering Analysis ---")
    count_in_split = 0
    count_in_split_matching_cat = 0
    lengths_in_split_matching_cat = []

    for record in full_idx:
        if record["paper_id"] in paper_ids_for_split:
            count_in_split += 1
            # Check category match
            category_match = (target_cat_set is None) or (record.get("cat") in target_cat_set)
            if category_match:
                count_in_split_matching_cat += 1
                lengths_in_split_matching_cat.append(record.get("length", 0))
                category_docs_in_split.append(record) # Also store record for potential sample print

    print(f"Total documents found in index file matching '{args.split}' split IDs: {count_in_split}")
    if target_cat_set:
        print(f"Documents matching category '{args.target_category}' in '{args.split}' split: {count_in_split_matching_cat}")
        if lengths_in_split_matching_cat:
            lengths_np = np.array(lengths_in_split_matching_cat)
            print(f"  Lengths (min/max/mean): {np.min(lengths_np):,} / {np.max(lengths_np):,} / {np.mean(lengths_np):,.2f}")
            num_long_enough = sum(1 for l in lengths_np if l >= min_required_length)
            print(f"  Number potentially suitable (length >= {min_required_length}): {num_long_enough}")
        else:
            print("  No documents found for this category in this split.")
    else:
         print(f"Analyzing all categories within the split.")
         if lengths_in_split_matching_cat:
             lengths_np = np.array(lengths_in_split_matching_cat)
             print(f"  Lengths (min/max/mean): {np.min(lengths_np):,} / {np.max(lengths_np):,} / {np.mean(lengths_np):,.2f}")
             num_long_enough = sum(1 for l in lengths_np if l >= min_required_length)
             print(f"  Number potentially suitable (length >= {min_required_length}): {num_long_enough}")


    print("-" * 20)
    print("Instantiating RandomWindowDataset...")

    target_categories_list = [args.target_category] if args.target_category else None
    try:
        # Instantiate the dataset - it will redo the filtering internally
        dataset = RandomWindowDataset(
            preprocessed_dir=data_dir, # Use resolved path
            split=args.split,
            target_categories=target_categories_list,
            sequence_length=args.sequence_length
        )

        print("-" * 20)
        print(f"--- Dataset Initialization Result ---")
        print(f"Number of documents in dataset pool (len(dataset.pool) -> dataset.__len__()): {len(dataset)}") # Use len(dataset) which calls __len__

        if dataset.pool:
            pool_lengths = [r['length'] for r in dataset.pool]
            print(f"Lengths in Final Pool (min/max/mean): {np.min(pool_lengths):,} / {np.max(pool_lengths):,} / {np.mean(pool_lengths):,.2f}")
            print(f"Confirming all pool lengths >= {min_required_length}: {all(l >= min_required_length for l in pool_lengths)}")
            print("\nSample documents in pool (first 5):")
            for i, record in enumerate(dataset.pool[:5]):
                print(f"  {i+1}: ID={record['paper_id']}, Cat={record['cat']}, Length={record['length']}, Offset={record['offset']}")
        else:
            print("Dataset pool is empty.")

    except FileNotFoundError:
        print(f"ERROR: Preprocessed data not found in {data_dir} during dataset init.")
    except ValueError as e:
        print(f"ERROR during dataset initialization: {e}")
    except Exception as e:
        print(f"UNEXPECTED ERROR during dataset initialization: {e}", exc_info=True) # Show traceback

if __name__ == "__main__":
    main() 