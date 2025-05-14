import os
from pathlib import Path
import datasets
from tqdm import tqdm

# Define the project root and output path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "tokenizer_training_corpus.txt"

# The 5 selected categories
SELECTED_CATEGORIES = ["cs.DS", "math.ST", "math.GR", "cs.IT", "cs.PL"]
TEXT_COLUMN = "text"  # Changed from "abstract" to "text"

def main():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)

    print("Loading ccdv/arxiv-classification dataset...")
    dataset_successfully_loaded = False
    dataset_uses_text_categories = False # Flag to indicate structure
    
    try:
        # Try loading default config first, which might have text categories
        full_dataset = datasets.load_dataset("ccdv/arxiv-classification", trust_remote_code=True)
        # Check if 'categories' field exists and is suitable (e.g., a list of strings)
        # We'll check the first example of the 'train' split if available
        if "train" in full_dataset and len(full_dataset["train"]) > 0:
            example_features = full_dataset["train"].features
            if "categories" in example_features and isinstance(example_features["categories"], datasets.features.Sequence):
                 # Further check if the inner feature is a Value feature of string type
                if isinstance(example_features["categories"].feature, datasets.features.Value) and example_features["categories"].feature.dtype == 'string':
                    dataset_uses_text_categories = True
                    print("Dataset loaded with default config, using 'categories' field (list of strings).")
                elif isinstance(example_features["categories"].feature, datasets.features.ClassLabel):
                     # This case means 'categories' is a ClassLabel, like 'label' in 'no_ref'
                     # For simplicity, we'll treat this like 'no_ref' and use the label mapping logic later
                     print("Dataset loaded with default config, 'categories' is ClassLabel. Will use label mapping.")
                     # We'll need to use the label mapping for this config too
                else:
                    print("Dataset loaded with default config, but 'categories' field is not a list of strings or ClassLabel. Fallback might be needed or logic adjusted.")
            elif "label" in example_features: # If no 'categories', but 'label' exists
                 print("Dataset loaded with default config, using 'label' field. Will use label mapping.")
            else: # Neither 'categories' (as list of strings) nor 'label' found in default
                print("Default config loaded, but required category fields ('categories' as list of strings or 'label') not found. Attempting fallback.")
                # This will force a fallback if the structure isn't what we expect
                raise ValueError("Default config structure not as expected for categories.")

        dataset_successfully_loaded = True
    except Exception as e:
        print(f"Error loading dataset with default config: {e}")
        print("Attempting to load with specific version 'no_ref' and split 'train' as a fallback strategy...")
        try:
            # Fallback to 'no_ref'
            full_dataset = datasets.DatasetDict({
                "train": datasets.load_dataset("ccdv/arxiv-classification", "no_ref", split="train", trust_remote_code=True)
            })
            # 'no_ref' definitely uses 'label' and requires mapping
            dataset_uses_text_categories = False 
            print("Fallback successful. Using only 'train' split from 'no_ref' config (uses 'label' field).")
            dataset_successfully_loaded = True
        except Exception as e_fallback:
            print(f"Fallback dataset loading failed: {e_fallback}")
            return

    if not dataset_successfully_loaded:
        print("Failed to load dataset in any known configuration.")
        return
    
    print("Dataset loaded. Aggregating text from selected categories...")
    all_texts = []

    for split_name in full_dataset.keys():
        print(f"Processing split: {split_name}")
        current_split_dataset = full_dataset[split_name]
        
        # Get label names if we need to map from numerical labels
        label_names_list = None
        if not dataset_uses_text_categories and 'label' in current_split_dataset.features and hasattr(current_split_dataset.features['label'], 'names'):
            label_names_list = current_split_dataset.features['label'].names
        elif dataset_uses_text_categories and 'categories' in current_split_dataset.features and isinstance(current_split_dataset.features['categories'].feature, datasets.features.ClassLabel):
            # This handles the case where default config has 'categories' as ClassLabel
            label_names_list = current_split_dataset.features['categories'].names


        def filter_function(example):
            document_category_full_name = None
            if dataset_uses_text_categories and 'categories' in example and isinstance(example['categories'], list):
                # Default config with 'categories' as list of strings
                # This case means example['categories'] is already a list of full text category strings.
                # We need to check if any of our short codes are part of any of these full names.
                for full_cat_name_in_doc in example['categories']:
                    if any(short_code in full_cat_name_in_doc for short_code in SELECTED_CATEGORIES):
                        return True
                return False
            elif label_names_list and 'label' in example:
                # 'no_ref' config (or default that behaves like it) using 'label' (integer)
                label_idx = example['label']
                if 0 <= label_idx < len(label_names_list):
                    document_category_full_name = label_names_list[label_idx]
            elif label_names_list and 'categories' in example and isinstance(current_split_dataset.features['categories'].feature, datasets.features.ClassLabel):
                # Default config where 'categories' is a ClassLabel (integer)
                label_idx = example['categories'] # This assumes 'categories' field holds the integer index
                if 0 <= label_idx < len(label_names_list):
                    document_category_full_name = label_names_list[label_idx]
            
            if document_category_full_name:
                return any(short_code in document_category_full_name for short_code in SELECTED_CATEGORIES)
            
            return False

        print(f"Filtering for categories: {SELECTED_CATEGORIES}...")
        filtered_dataset = current_split_dataset.filter(
            filter_function,
            num_proc=os.cpu_count()
        )
        print(f"Found {len(filtered_dataset)} documents in split '{split_name}' for the selected categories.")

        missing_text_count = 0
        for i, doc in enumerate(tqdm(filtered_dataset, desc=f"Extracting text from {split_name}")):
            if TEXT_COLUMN in doc and doc[TEXT_COLUMN]:
                all_texts.append(doc[TEXT_COLUMN].replace("\\n", " ").replace("\\r", " ")) # Normalize newlines
            else:
                missing_text_count += 1
                if missing_text_count < 5: # Log first few missing cases
                    print(f"Warning: Document {i} in split '{split_name}' is missing '{TEXT_COLUMN}' or it is empty.")
                    # print(f"Full doc (missing text): {doc}") # Careful: this might be very verbose
        
        if missing_text_count > 0:
            print(f"Info: In split '{split_name}', {missing_text_count} documents were filtered but had no/empty '{TEXT_COLUMN}'.")

    print(f"Total documents aggregated from all splits: {len(all_texts)}")

    if not all_texts:
        print("No text data found for the selected categories. Exiting.")
        return

    print(f"Saving aggregated text to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for text in tqdm(all_texts, desc="Writing to file"):
            f.write(text + "\\n")

    print(f"Successfully saved {len(all_texts)} documents to {OUTPUT_FILE}")
    print(f"Total lines in output file: {len(all_texts)}")
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Output file size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main() 