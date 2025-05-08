import os
import json
import argparse # Added argparse
from pathlib import Path
import numpy as np
# import re # Removed, no longer needed
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import logging # Added logging
import boto3 # Added boto3
from botocore.exceptions import ClientError # Added for S3 error handling
from typing import Optional # Added Optional for type hint
from dotenv import load_dotenv # Added dotenv

# Define the dataset name and configuration
RAW_DATASET_NAME = "ccdv/arxiv-classification"
DATASET_CONFIG = "no_ref" # Use the built-in config to remove refs
RAW_DATA_CACHE_DIR_NAME = "data_cache_raw_arxiv"

# Define the output directory for preprocessed data
PREPROCESSED_DIR_NAME = "preprocessed_arxiv"

# Model for tokenizer
TOKENIZER_NAME = "EleutherAI/pythia-70m-deduped" # As per EXPERIMENT_PLAN.md

DEFAULT_MAX_EXAMPLES_PER_SPLIT = 50 # Default limit for quick testing

OUTPUT_FILES = ["tokens.bin", "index.jsonl", "splits.json"]

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def download_from_s3(bucket_name: str, prefix: str, local_dir: Path):
    """Attempts to download required files from S3."""
    s3 = boto3.client('s3')
    logging.info(f"Attempting to download preprocessed data from s3://{bucket_name}/{prefix}")
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    all_successful = True
    for filename in OUTPUT_FILES:
        s3_key = f"{prefix.rstrip('/')}/{filename}"
        local_path = local_dir / filename
        try:
            logging.info(f"Downloading {s3_key} to {local_path}...")
            s3.download_file(bucket_name, s3_key, str(local_path))
            logging.info(f"Successfully downloaded {filename}.")
        except ClientError as e:
            logging.warning(f"Failed to download {s3_key} from S3: {e}")
            all_successful = False
            # Clean up partially downloaded file if it exists
            if local_path.exists():
                local_path.unlink()
            break # Stop trying if one file fails
            
    if not all_successful:
         logging.warning("Could not download all required files from S3. Proceeding with local processing.")
         # Clean up directory if partially created
         if local_dir.exists() and not any(local_dir.iterdir()):
             local_dir.rmdir()
         elif local_dir.exists(): # Clean up any files downloaded before failure
             for filename in OUTPUT_FILES:
                 if (local_dir / filename).exists():
                     (local_dir / filename).unlink()
         return False
    
    logging.info(f"Successfully downloaded all preprocessed data from s3://{bucket_name}/{prefix}")
    return True

def upload_to_s3(bucket_name: str, prefix: str, local_dir: Path):
    """Uploads processed files to S3."""
    s3 = boto3.client('s3')
    logging.info(f"Attempting to upload preprocessed data to s3://{bucket_name}/{prefix}")
    
    all_successful = True
    for filename in OUTPUT_FILES:
        s3_key = f"{prefix.rstrip('/')}/{filename}"
        local_path = local_dir / filename
        if not local_path.exists():
            logging.error(f"Local file {local_path} not found for upload.")
            all_successful = False
            break
        try:
            logging.info(f"Uploading {local_path} to {s3_key}...")
            s3.upload_file(str(local_path), bucket_name, s3_key)
            logging.info(f"Successfully uploaded {filename}.")
        except ClientError as e:
            logging.error(f"Failed to upload {local_path} to S3: {e}")
            all_successful = False
            # Decide if we want to break or continue trying other files
            break 
            
    if not all_successful:
        logging.error("Failed to upload all required files to S3.")
    else:
        logging.info(f"Successfully uploaded all preprocessed data to s3://{bucket_name}/{prefix}")

def fetch_tokenize_and_save(max_examples: Optional[int], # Now takes Optional[int]
                              output_dir: Path,
                              raw_cache_dir: Path):
    """
    Downloads the ccdv/arxiv-classification dataset, tokenizes it,
    and saves it into a memory-mappable binary file with JSON indexes.
    Optionally limits the number of examples per split.
    """
    # 1. Create directories if they don't exist (might exist if S3 download failed)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Attempting to load raw dataset: {RAW_DATASET_NAME} (config: {DATASET_CONFIG})...")
    dataset_dict = load_dataset(
        RAW_DATASET_NAME, 
        DATASET_CONFIG, 
        cache_dir=str(raw_cache_dir), 
        trust_remote_code=True
    )
    logging.info(f"Raw dataset loaded. Splits: {list(dataset_dict.keys())}")

    logging.info(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Common practice if pad_token is not set
    
    # Get category names from dataset features (and clean them)
    # e.g. 'math.AC\n' -> 'math.AC'
    category_names_raw = dataset_dict['train'].features['label'].names
    category_names_cleaned = [name.strip() for name in category_names_raw]
    
    # Prepare regex for removing category references from text - REMOVED
    # category_ref_patterns = [r"\[{}\]".format(re.escape(cat)) for cat in category_names_cleaned]
    # combined_category_regex = re.compile("|".join(category_ref_patterns))

    master_token_list = []
    index_data = []
    splits_data = {split: [] for split in dataset_dict.keys()}
    current_offset = 0

    for split_name in dataset_dict.keys(): 
        logging.info(f"Processing split: {split_name}...")
        dataset_split = dataset_dict[split_name]
        
        num_processed_in_split = 0
        iterator = tqdm(dataset_split, desc=f"Tokenizing {split_name}")
        for i, paper_data in enumerate(iterator):
            # Apply limit unless max_examples is None (which means --full_dataset was passed)
            if max_examples is not None and num_processed_in_split >= max_examples:
                logging.info(f"Reached max_examples ({max_examples}) for split '{split_name}'. Moving to next split.")
                break 

            text = paper_data['text']
            label_id = paper_data['label']
            category = category_names_cleaned[label_id]
            paper_id = f"{split_name}_{i}" # Simple unique ID based on split and original index

            # Skip if text is None or empty
            if not text:
                logging.warning(f"Skipping paper {paper_id} due to empty text.")
                continue

            # Clean text: Remove in-text category references like [cs.CV] - REMOVED
            # cleaned_text = combined_category_regex.sub("", text)

            # Tokenize
            try:
                tokens = tokenizer(text, truncation=False)["input_ids"]
            except Exception as e:
                 logging.warning(f"Skipping paper {paper_id} due to tokenization error: {e}")
                 continue # Skip this paper if tokenization fails

            if not tokens:
                logging.warning(f"Skipping paper {paper_id} as tokenization resulted in empty sequence.")
                continue
                
            num_tokens = len(tokens)
            
            # Append to master list (convert to uint16 here)
            master_token_list.extend(np.array(tokens, dtype=np.uint16))
            
            # Add entry to index
            index_data.append({
                "paper_id": paper_id,
                "cat": category,
                "offset": current_offset,
                "length": num_tokens
            })
            
            # Add paper ID to the current split
            splits_data[split_name].append(paper_id)
            
            # Update offset
            current_offset += num_tokens
            num_processed_in_split += 1

    logging.info(f"Total tokens processed: {current_offset}")

    # 3. Save processed data locally
    logging.info("Saving processed data locally...")
    
    # Save tokens.bin
    tokens_bin_path = output_dir / "tokens.bin"
    logging.info(f"Saving tokens to {tokens_bin_path}...")
    np.array(master_token_list, dtype=np.uint16).tofile(tokens_bin_path)
    logging.info(f"Saved {tokens_bin_path}.")

    # Save index.jsonl
    index_jsonl_path = output_dir / "index.jsonl"
    logging.info(f"Saving index to {index_jsonl_path}...")
    with open(index_jsonl_path, "w") as f:
        for row in index_data:
            f.write(json.dumps(row) + "\n")
    logging.info(f"Saved {index_jsonl_path}.")

    # Save splits.json
    splits_json_path = output_dir / "splits.json"
    logging.info(f"Saving splits map to {splits_json_path}...")
    with open(splits_json_path, "w") as f:
        json.dump(splits_data, f, indent=4)
    logging.info(f"Saved {splits_json_path}.")
    
    logging.info(f"All local preprocessing complete. Output in {output_dir}")

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    # This needs to happen before boto3.client() is called implicitly or explicitly
    # Boto3 checks environment variables during client creation.
    load_dotenv()
    logging.info(".env file loaded if present.")

    parser = argparse.ArgumentParser(description="Download, tokenize, and save the ArXiv classification dataset.")
    parser.add_argument("--output_dir", type=Path, default=Path(f"./{PREPROCESSED_DIR_NAME}"), 
                        help=f"Directory to save preprocessed output ({PREPROCESSED_DIR_NAME}).")
    parser.add_argument("--raw_cache_dir", type=Path, default=Path(f"./{RAW_DATA_CACHE_DIR_NAME}"), 
                        help=f"Directory to cache raw Hugging Face dataset ({RAW_DATA_CACHE_DIR_NAME}).")
    parser.add_argument("--max_examples_per_split", type=int, default=DEFAULT_MAX_EXAMPLES_PER_SPLIT, 
                        help="Maximum number of examples to process per split (train/val/test). Set to 0 or negative for no limit if --full_dataset is not used.")
    parser.add_argument("--full_dataset", action="store_true", 
                        help="Process the full dataset, ignoring --max_examples_per_split.")
    
    # S3 Arguments
    parser.add_argument("--download_from_s3", action="store_true",
                        help="Attempt to download preprocessed data from S3 before processing locally.")
    parser.add_argument("--upload_to_s3", action="store_true",
                        help="Upload preprocessed data to S3 after local processing.")
    parser.add_argument("--s3_bucket", type=str, default=None,
                        help="S3 bucket name for download/upload.")
    parser.add_argument("--s3_prefix", type=str, default=PREPROCESSED_DIR_NAME,
                        help=f"Prefix (folder path) within the S3 bucket. Defaults to '{PREPROCESSED_DIR_NAME}'.")

    args = parser.parse_args()

    # Determine max examples limit
    if args.full_dataset:
        max_examples = None
        logging.info("Processing full dataset (--full_dataset specified).")
    elif args.max_examples_per_split <= 0:
        max_examples = None
        logging.info(f"Processing full dataset (max_examples_per_split={args.max_examples_per_split}).")
    else:
        max_examples = args.max_examples_per_split
        logging.info(f"Processing limited dataset (max_examples_per_split={max_examples}).")

    # S3 Download Logic
    downloaded_ok = False
    if args.download_from_s3:
        if not args.s3_bucket:
            logging.error("S3 bucket name must be provided using --s3_bucket to download from S3.")
        else:
            downloaded_ok = download_from_s3(args.s3_bucket, args.s3_prefix, args.output_dir)

    # Main Processing Logic
    if not downloaded_ok:
        logging.info("Proceeding with local data fetch and processing...")
        try:
            fetch_tokenize_and_save(
                max_examples=max_examples, 
                output_dir=args.output_dir, 
                raw_cache_dir=args.raw_cache_dir
            )
            # S3 Upload Logic (only if local processing was successful)
            if args.upload_to_s3:
                if not args.s3_bucket:
                    logging.error("S3 bucket name must be provided using --s3_bucket to upload to S3.")
                else:
                    upload_to_s3(args.s3_bucket, args.s3_prefix, args.output_dir)
        except Exception as e:
            logging.exception(f"An error occurred during local processing: {e}") # Log full traceback
            # Optionally, decide if you want to exit with error code
            # sys.exit(1)
    
    logging.info("Script finished.")

    # Inspection block (remains unchanged)
    print("\n--- Dataset Inspection (Post-processing) ---")
    try: 
        dataset_dict_inspect = load_dataset(
            RAW_DATASET_NAME, 
            DATASET_CONFIG,
            cache_dir=str(args.raw_cache_dir), 
            trust_remote_code=True
        )
        if 'train' in dataset_dict_inspect:
            train_split = dataset_dict_inspect['train']
            print(f"Train split features: {train_split.features}")
            print(f"Number of examples in train split: {len(train_split)}")
            print(f"First example:\n{train_split[0]}")
        else:
            print("'train' split not found in the loaded dataset.")
    except Exception as e:
        print(f"Error during final dataset inspection: {e}")

    # --- Re-enable inspection code block, using the dataset already loaded ----
    # This requires the dataset_dict to be available outside fetch_tokenize_and_save
    # or re-loaded here. Re-loading is simpler for now.
    print("\n--- Dataset Inspection (Re-loading with 'no_ref' config for inspection) ---")
    try: 
        # Re-load the dataset specifically for inspection using the correct config
        dataset_dict_inspect = load_dataset(
            RAW_DATASET_NAME, 
            DATASET_CONFIG,
            cache_dir=str(args.raw_cache_dir), 
            trust_remote_code=True
        )
        
        if 'train' in dataset_dict_inspect:
            train_split = dataset_dict_inspect['train']
            print(f"\nFeatures of the 'train' split: {train_split.features}")
            if len(train_split) > 0:
                print(f"\nFirst example of the 'train' split (using '{DATASET_CONFIG}' config): \n{train_split[0]}")
            else:
                print("\n'train' split is empty.")
        else:
            print("\n'train' split not found in the dataset.")
    except Exception as e:
        print(f"Could not re-load dataset for inspection: {e}")
    print("--- End of Inspection ---") 