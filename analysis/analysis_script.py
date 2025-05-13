# analysis_script.py
# This script will be used for the main analysis workflow.
# It is structured for use with VS Code cells (# %%).

# %% 
# Example cell
# print("Hello from analysis_script.py") 

# %% Store Loss Profiles
# import json # Already imported, but good for cell context
# import os   # Already imported, but good for cell context

def store_loss_profiles(loss_profiles_map, base_output_dir='analysis/results'):
    """
    Stores loss profiles for one or more models to JSON files.

    Args:
        loss_profiles_map (dict): A dictionary where keys are model identifiers (e.g., run prefix or name)
                                  and values are the list of per-sequence loss profiles 
                                  (each profile being a list/tensor of per-token losses).
        base_output_dir (str): The base directory to save the JSON files.
    Returns:
        dict: A dictionary mapping model_identifier to the path of the stored JSON file.
    """
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Storing loss profiles in: {base_output_dir}")
    stored_paths = {}

    for model_identifier, profiles in loss_profiles_map.items():
        safe_model_name = model_identifier.replace('/', '_').replace(':', '_')
        file_path = os.path.join(base_output_dir, f"{safe_model_name}_loss_profiles.json")
        
        profiles_to_save = []
        if profiles:
            for profile in profiles:
                if hasattr(profile, 'tolist'): 
                    profiles_to_save.append(profile.tolist())
                else:
                    profiles_to_save.append(profile) 
        
        try:
            with open(file_path, 'w') as f:
                json.dump(profiles_to_save, f, indent=4)
            print(f"  Successfully stored loss profiles for '{model_identifier}' at: {file_path}")
            stored_paths[model_identifier] = file_path
        except Exception as e:
            print(f"  Error storing loss profiles for '{model_identifier}' at {file_path}: {e}")
    return stored_paths # Ensure this return statement is present

# %% Plotting Utility for Loss Profiles
import matplotlib.pyplot as plt
import numpy as np
import json # Already imported
import os   # Already imported

def plot_average_loss_profile(
    loss_profiles_file_path,
    model_identifier, 
    output_dir='analysis/plots',
    max_seq_len_to_plot=None 
):
    """
    Loads loss profiles from a JSON file, calculates the average loss at each token position,
    and plots this average loss. Saves the plot to a file.

    Args:
        loss_profiles_file_path (str): Path to the JSON file containing loss profiles.
        model_identifier (str): A name or identifier for the model (e.g., run prefix), 
                                used in plot titles and output filenames.
        output_dir (str): Directory to save the plot image.
        max_seq_len_to_plot (int, optional): If specified, only plot losses up to this token position.
    Returns:
        str: The path to the saved plot file, or None if plotting failed.
    """
    print(f"Plotting average loss profile for '{model_identifier}' from: {loss_profiles_file_path}")
    os.makedirs(output_dir, exist_ok=True)
    plot_output_file_path = None 

    try:
        with open(loss_profiles_file_path, 'r') as f:
            loss_profiles = json.load(f) 
    except FileNotFoundError:
        print(f"  Error: Loss profile file not found: {loss_profiles_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON from: {loss_profiles_file_path}")
        return None
    except Exception as e:
        print(f"  Error loading loss profiles from {loss_profiles_file_path}: {e}")
        return None

    if not loss_profiles or not isinstance(loss_profiles, list) or not all(isinstance(p, list) for p in loss_profiles):
        print("  Error: Loss profiles data is empty, not a list, or not a list of lists.")
        return None
    
    valid_profiles = [p for p in loss_profiles if p] 
    if not valid_profiles:
        print("  Error: No valid (non-empty) loss profiles found to plot.")
        return None

    current_max_len = len(valid_profiles[0])
    if max_seq_len_to_plot is not None:
        plot_len = min(current_max_len, max_seq_len_to_plot)
    else:
        plot_len = current_max_len

    profiles_for_avg = [p[:plot_len] for p in valid_profiles if len(p) >= plot_len]

    if not profiles_for_avg:
        print(f"  Error: No profiles are long enough (>= {plot_len} tokens) for averaging.")
        return None

    try:
        avg_losses = np.mean(profiles_for_avg, axis=0)
    except Exception as e:
        print(f"  Error calculating mean of loss profiles: {e}")
        return None

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, plot_len + 1), avg_losses) 
    plt.title(f'Average In-Context Loss vs. Token Position\nModel: {model_identifier}')
    plt.xlabel('Token Position in Sequence (Context Length + 1)')
    plt.ylabel('Average Cross-Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()

    safe_model_name_plot = model_identifier.replace('/', '_').replace(':', '_')
    plot_file_name = f"{safe_model_name_plot}_avg_loss_vs_position.png"
    plot_output_file_path = os.path.join(output_dir, plot_file_name)
    
    try:
        plt.savefig(plot_output_file_path)
        print(f"  Plot saved to: {plot_output_file_path}")
    except Exception as e:
        print(f"  Error saving plot to {plot_output_file_path}: {e}")
        plot_output_file_path = None 
    finally:
        plt.close() 
    
    return plot_output_file_path # Ensure this return statement is present

# %% List S3 Training Runs
import boto3

def list_s3_training_runs(bucket_name, orchestrator_prefix):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    subfolders = []
    # Ensure orchestrator_prefix ends with a '/' for correct prefix matching
    if not orchestrator_prefix.endswith('/'):
        orchestrator_prefix += '/'

    for result in paginator.paginate(Bucket=bucket_name, Prefix=orchestrator_prefix, Delimiter='/'):
        for common_prefix in result.get('CommonPrefixes', []):
            subfolders.append(common_prefix.get('Prefix'))
    return subfolders

# Example usage (uncomment to test):
# BUCKET_NAME = 'obelisk-simplex'
# ORCHESTRATOR_PATH = 'non-ergodic-arxiv/training_runs/run_2025-05-11_05-33-44_final_python_orch/' 
# # Note: Ensure the ORCHESTRATOR_PATH above is an actual path that exists and contains subfolders for testing.
# training_runs = list_s3_training_runs(BUCKET_NAME, ORCHESTRATOR_PATH)
# print(f"Found {len(training_runs)} training runs:")
# for run in training_runs:
#     print(run)

# %%
# %% Get Training Run Metadata
# import boto3 # Already imported in the previous cell
import tempfile
import json
import os
import re # For parsing log files
import ast # For safely evaluating string representation of list

def get_train_run_metadata(bucket_name, run_prefixes):
    s3_client = boto3.client('s3')
    metadata_list = []
    model_weight_files = ["model.safetensors", "pytorch_model.bin"]
    # Refined regex to specifically capture the list structure
    category_log_pattern = re.compile(r"Selected \d+ categories for training \(seed \d+\): (\[.*\])")

    for run_prefix in run_prefixes:
        current_run_meta = {
            'prefix': run_prefix,
            'model_config_s3_path': None,
            'model_weights_s3_path': None,
            'model_source_type': None, # To log where the model was found
            'args_s3_path': None,
            'args_content': None, 
            'log_s3_path': None,
            'training_categories': None, 
            'checkpoints': [] # For general checkpoints, not used for final model selection
        }

        # Fetch args_content and training_categories first as they are independent of model choice
        try:
            objects_in_run_prefix = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=run_prefix, Delimiter='/')
            
            # Args.json processing (copied from original, ensure it runs early)
            if 'Contents' in objects_in_run_prefix:
                for obj in objects_in_run_prefix.get('Contents', []):
                    file_key = obj['Key']
                    if file_key.endswith('_args.json'):
                        current_run_meta['args_s3_path'] = f"s3://{bucket_name}/{file_key}"
                        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_json_file_obj:
                            tmp_json_file_name = tmp_json_file_obj.name
                        try:
                            s3_client.download_file(bucket_name, file_key, tmp_json_file_name)
                            if os.path.getsize(tmp_json_file_name) > 0:
                                with open(tmp_json_file_name, "r") as f_json:
                                    current_run_meta['args_content'] = json.load(f_json)
                            else:
                                print(f"Warning: Downloaded {file_key} is empty.")
                                current_run_meta['args_content'] = "<EMPTY_ON_S3_OR_DOWNLOAD_FAILED>"
                        except Exception as e:
                            print(f"Error processing _args.json {file_key}: {e}")
                            current_run_meta['args_content'] = f"<ERROR_PROCESSING: {e}>"
                        finally:
                            if os.path.exists(tmp_json_file_name):
                                os.remove(tmp_json_file_name)
                        break 
            
            # Log file processing (copied from original, ensure it runs early)
            if 'Contents' in objects_in_run_prefix:
                for obj in objects_in_run_prefix.get('Contents', []):
                    file_key = obj['Key']
                    if file_key.endswith('.log'):
                        current_run_meta['log_s3_path'] = f"s3://{bucket_name}/{file_key}"
                        with tempfile.NamedTemporaryFile(mode="w", encoding='utf-8', delete=False) as tmp_log_file_obj:
                            tmp_log_file_name = tmp_log_file_obj.name
                        try:
                            s3_client.download_file(bucket_name, file_key, tmp_log_file_name)
                            if os.path.getsize(tmp_log_file_name) > 0:
                                with open(tmp_log_file_name, "r", encoding='utf-8', errors='replace') as f_log:
                                    for line in f_log:
                                        match = category_log_pattern.search(line)
                                        if match:
                                            try:
                                                categories_str = match.group(1) 
                                                current_run_meta['training_categories'] = ast.literal_eval(categories_str)
                                                break 
                                            except (SyntaxError, ValueError) as e_parse:
                                                print(f"Error parsing categories from log line '{line.strip()}' in {file_key}: {e_parse}")
                                                current_run_meta['training_categories'] = f"<PARSE_ERROR: {e_parse}>"
                                                break
                            else:
                                print(f"Warning: Downloaded {file_key} is empty.")
                                current_run_meta['training_categories'] = "<EMPTY_LOG_OR_DOWNLOAD_FAILED>"
                        except Exception as e:
                            print(f"Error processing .log file {file_key}: {e}")
                            current_run_meta['training_categories'] = f"<ERROR_PROCESSING_LOG: {e}>"
                        finally:
                            if os.path.exists(tmp_log_file_name):
                                os.remove(tmp_log_file_name)
                        break
        except Exception as e:
            print(f"Error listing initial objects for {run_prefix}: {e}")


        # --- Model Finding Logic ---
        chosen_model_base_s3_prefix = None # S3 prefix for the directory of the chosen model (e.g., .../loss_checkpoints/loss_X_stepY/)
        model_found_in_chosen_location = False

        # 1. Try 'loss_checkpoints/'
        loss_checkpoints_s3_prefix = run_prefix + "loss_checkpoints/"
        print(f"  Attempting to find loss checkpoints in: {loss_checkpoints_s3_prefix}")
        loss_checkpoints_objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=loss_checkpoints_s3_prefix, Delimiter='/')
        
        candidate_loss_checkpoints = []
        if 'CommonPrefixes' in loss_checkpoints_objects:
            print(f"    Found CommonPrefixes in {loss_checkpoints_s3_prefix}: {loss_checkpoints_objects.get('CommonPrefixes')}")
            for common_prefix_obj in loss_checkpoints_objects.get('CommonPrefixes', []):
                checkpoint_folder_s3_prefix_from_list = common_prefix_obj.get('Prefix')
                folder_name = checkpoint_folder_s3_prefix_from_list.split('/')[-2]
                print(f"      Processing loss checkpoint folder: {folder_name} (full prefix: {checkpoint_folder_s3_prefix_from_list})")
                loss_checkpoint_folder_pattern = re.compile(r"loss_(\d+\.\d{1,})_step(\d{6,})")
                match = loss_checkpoint_folder_pattern.fullmatch(folder_name)
                if match:
                    try:
                        loss_val = float(match.group(1))
                        step_val = int(match.group(2))
                        print(f"        Regex match successful: loss={loss_val}, step={step_val}")
                        candidate_loss_checkpoints.append({'loss': loss_val, 'prefix': checkpoint_folder_s3_prefix_from_list, 'step': step_val})
                    except ValueError:
                        print(f"        Warning: Could not parse float loss or int step from checkpoint folder: {folder_name} in {run_prefix}. Match groups: {match.groups()}")
                else:
                    print(f"        Regex match failed for folder name: {folder_name}")
        else:
            print(f"    No CommonPrefixes (subfolders) found directly under {loss_checkpoints_s3_prefix}")
        
        print(f"    Candidate loss checkpoints before sorting: {candidate_loss_checkpoints}")
        if candidate_loss_checkpoints:
            candidate_loss_checkpoints.sort(key=lambda x: x['loss'])
            print(f"    Candidate loss checkpoints after sorting by loss: {candidate_loss_checkpoints}")
            best_loss_checkpoint = candidate_loss_checkpoints[0]
            chosen_model_base_s3_prefix = best_loss_checkpoint['prefix'] # This is the S3 "folder" path
            current_run_meta['model_source_type'] = f"loss_checkpoint (loss: {best_loss_checkpoint['loss']:.4f}, step: {best_loss_checkpoint.get('step', 'N/A')})"
            print(f"  Selected best loss checkpoint for {run_prefix}: {chosen_model_base_s3_prefix} with loss {best_loss_checkpoint['loss']:.4f}")

            # Now, find config and weights WITHIN this chosen_model_base_s3_prefix
            try:
                response_in_chosen_ckpt = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=chosen_model_base_s3_prefix)
                if 'Contents' in response_in_chosen_ckpt:
                    print(f"    Files in chosen loss checkpoint ({chosen_model_base_s3_prefix}):")
                    for obj_in_ckpt in response_in_chosen_ckpt['Contents']:
                        file_key_in_ckpt = obj_in_ckpt['Key']
                        print(f"      - {file_key_in_ckpt}")
                        # Ensure we match 'config.json' specifically, not 'generation_config.json' etc.
                        if os.path.basename(file_key_in_ckpt) == 'config.json':
                            current_run_meta['model_config_s3_path'] = f"s3://{bucket_name}/{file_key_in_ckpt}"
                        for weight_file_name in model_weight_files:
                            if file_key_in_ckpt.endswith(weight_file_name): # endswith is fine for diverse weight names
                                current_run_meta['model_weights_s3_path'] = f"s3://{bucket_name}/{file_key_in_ckpt}"
                                break # Found one weight file, break from inner loop
                        # Check if both found to make model_found_in_chosen_location true
                        if current_run_meta['model_config_s3_path'] and current_run_meta['model_weights_s3_path']:
                            model_found_in_chosen_location = True
                            print(f"    Found config and weights in chosen loss checkpoint: {chosen_model_base_s3_prefix}")
                            break # Found both, break from outer loop (iterating files in chosen_model_base_s3_prefix)
                if not model_found_in_chosen_location:
                     print(f"    Warning: Model config or weights not found within the chosen loss checkpoint: {chosen_model_base_s3_prefix}")
            except Exception as e_list_in_ckpt:
                print(f"    Error listing objects within chosen loss checkpoint {chosen_model_base_s3_prefix}: {e_list_in_ckpt}")
        else:
            print(f"  No valid loss checkpoints were parsed and added to candidates for {run_prefix}. Will try 'best_model/'.")

        # 2. Fallback to 'best_model/' if no loss checkpoint was chosen OR if files weren't found in the chosen one
        if not chosen_model_base_s3_prefix or not model_found_in_chosen_location:
            if not chosen_model_base_s3_prefix: # Only print this if loss_checkpoints was entirely skipped
                print(f"  No loss checkpoint selected. Falling back to 'best_model/' for {run_prefix}")
            else: # Printed if loss_checkpoint was selected but files were missing
                 print(f"  Files not found in selected loss checkpoint. Falling back to 'best_model/' for {run_prefix}")

            best_model_dir_prefix = run_prefix + 'best_model/'
            current_run_meta['model_source_type'] = "best_model" # Set source type for fallback
            try:
                response_best_model = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=best_model_dir_prefix)
                if 'Contents' in response_best_model:
                    for obj in response_best_model['Contents']:
                        file_key = obj['Key']
                        if file_key.endswith('config.json'):
                            current_run_meta['model_config_s3_path'] = f"s3://{bucket_name}/{file_key}"
                        for weight_file_name in model_weight_files:
                            if file_key.endswith(weight_file_name):
                                current_run_meta['model_weights_s3_path'] = f"s3://{bucket_name}/{file_key}"
                                break
                        if current_run_meta['model_config_s3_path'] and current_run_meta['model_weights_s3_path']:
                            print(f"    Found config and weights in 'best_model/' for {run_prefix}")
                            break 
                if not (current_run_meta['model_config_s3_path'] and current_run_meta['model_weights_s3_path']):
                    print(f"    Warning: Model config or weights not found in 'best_model/' for {run_prefix} either.")
            except Exception as e_best_model:
                print(f"    Error listing objects for {best_model_dir_prefix}: {e_best_model}")
        
        # General checkpoints listing (for informational purposes, not primary model selection)
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=run_prefix, Delimiter='/'):
                if 'CommonPrefixes' in page: 
                    for common_prefix_obj in page['CommonPrefixes']:
                        potential_checkpoint_folder_s3_prefix = common_prefix_obj.get('Prefix')
                        if potential_checkpoint_folder_s3_prefix and 'checkpoint-' in potential_checkpoint_folder_s3_prefix.rstrip('/').split('/')[-1]:
                            # This part remains unchanged as it's for listing general checkpoints, not selecting the primary model
                            checkpoint_data = {
                                's3_prefix': potential_checkpoint_folder_s3_prefix,
                                'config_s3_path': None,
                                'weights_s3_path': None
                            }
                            ckpt_files_response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=potential_checkpoint_folder_s3_prefix)
                            if 'Contents' in ckpt_files_response:
                                for ckpt_file_obj in ckpt_files_response['Contents']:
                                    ckpt_file_key = ckpt_file_obj['Key']
                                    if ckpt_file_key.endswith('config.json'):
                                        checkpoint_data['config_s3_path'] = f"s3://{bucket_name}/{ckpt_file_key}"
                                    for weight_file_name in model_weight_files:
                                        if ckpt_file_key.endswith(weight_file_name):
                                            checkpoint_data['weights_s3_path'] = f"s3://{bucket_name}/{ckpt_file_key}"
                                            break 
                            if checkpoint_data['config_s3_path'] or checkpoint_data['weights_s3_path']:
                                current_run_meta['checkpoints'].append(checkpoint_data)
        except Exception as e_gen_ckpt:
            print(f"Error listing/processing general checkpoints for {run_prefix}: {e_gen_ckpt}")
        
        metadata_list.append(current_run_meta)
    return metadata_list

# %%
# %% Download Training Run Files
# import os # Already imported
# import boto3 # Already imported

def get_train_run_model_files(bucket_name, s3_object_keys, local_base_download_path):
    s3_client = boto3.client('s3')
    downloaded_files_local_paths = []

    for s3_key in s3_object_keys:
        if not s3_key: # Skip if the S3 key is None or empty
            print(f"Skipping empty or None S3 key.")
            continue

        # Construct the full local path
        # The s3_key might be like 'non-ergodic-arxiv/training_runs/X/Y/file.json'
        # We want it under local_base_download_path/non-ergodic-arxiv/training_runs/X/Y/file.json
        local_file_path = os.path.join(local_base_download_path, s3_key)
        
        try:
            # Ensure the local directory exists
            local_file_dir = os.path.dirname(local_file_path)
            os.makedirs(local_file_dir, exist_ok=True)
            
            print(f"Attempting to download s3://{bucket_name}/{s3_key} to {local_file_path}...")
            s3_client.download_file(bucket_name, s3_key, local_file_path)
            
            if os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
                print(f"Successfully downloaded s3://{bucket_name}/{s3_key} to {local_file_path}")
                downloaded_files_local_paths.append(local_file_path)
            else:
                print(f"Error: File {local_file_path} was not downloaded correctly (missing or empty).")
        except Exception as e:
            print(f"Error downloading {s3_key} from bucket {bucket_name} to {local_file_path}: {e}")
            
    return downloaded_files_local_paths

# %%
# %% Load Hugging Face Model
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(local_model_dir_path, run_args_content):
    if not run_args_content or 'model_name_or_path' not in run_args_content:
        print(f"Error: 'model_name_or_path' not found in run_args_content.")
        return None, None

    base_tokenizer_path = run_args_content['model_name_or_path']
    print(f"Attempting to load model from: {local_model_dir_path}")
    print(f"Attempting to load tokenizer from base model: {base_tokenizer_path}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(local_model_dir_path)
        print(f"Successfully loaded model from {local_model_dir_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
        print(f"Successfully loaded tokenizer for {base_tokenizer_path}")
        
        # It's good practice to ensure the tokenizer's pad_token is set if using it for batching
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")
            
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model/tokenizer. Model path: {local_model_dir_path}, Tokenizer base: {base_tokenizer_path}. Error: {e}")
        return None, None

# %%
# %% Create Test Dataloader (using RandomWindowDataset)
from torch.utils.data import DataLoader
# Assuming RandomWindowDataset is in a file named random_window_dataset.py in the parent directory or accessible via sys.path
import sys
from pathlib import Path
import math # Added for ceil
PROJECT_ROOT_ANALYSIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT_ANALYSIS)) # Add project root for imports
from random_window_dataset import RandomWindowDataset, DEFAULT_PREPROCESSED_DIR

# CustomTextDataset class is removed

def create_test_dataloader(
    preprocessed_data_dir: Path,
    split_name: str, 
    tokenizer, # Keep for context, though RandomWindowDataset handles pre-tokenized data
    batch_size: int, 
    sequence_length: int, 
    target_categories: list = None, 
    eval_multiplier: int = 1 # This will be used directly by RandomWindowDataset
):
    print(f"Creating test dataloader using RandomWindowDataset from: {preprocessed_data_dir}")
    print(f"Split: {split_name}, Tokenizer context: {tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else 'unknown'}")
    print(f"Batch size: {batch_size}, Sequence length: {sequence_length}, Eval Multiplier: {eval_multiplier}")
    if target_categories:
        print(f"Target categories: {target_categories}")

    dataset = None

    try:
        dataset = RandomWindowDataset(
            preprocessed_dir=preprocessed_data_dir,
            split=split_name,
            target_categories=target_categories,
            sequence_length=sequence_length,
            eval_multiplier=eval_multiplier
        )
        print(f"  Dataset with eval_multiplier={eval_multiplier} created with {len(dataset)} samples.")

    except FileNotFoundError as e:
        print(f"Error: Preprocessed data not found. {e}")
        return None
    except ValueError as e: # Catches issues from RandomWindowDataset if categories are bad, etc.
        print(f"Error initializing RandomWindowDataset: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while initializing RandomWindowDataset: {e}")
        return None

    if dataset is None or len(dataset) == 0:
        print(f"Warning: The created dataset for split '{split_name}' (eval_multiplier: {eval_multiplier}) is empty. No batches will be produced.")
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=False # Typically False for test/validation sets
    )
    
    print(f"Successfully created test dataloader with {len(dataloader)} batches from {len(dataset)} samples (using eval_multiplier: {eval_multiplier}).")
    return dataloader

# %%
# %% Core Per-Token Cross-Entropy Loss Calculation
import torch
from torch.nn import CrossEntropyLoss

def calculate_per_token_loss(model, input_ids):
    """
    Calculates cross-entropy loss for each token in a batch, given its preceding context.

    Args:
        model: The loaded Hugging Face transformer model.
        input_ids (torch.Tensor): A batch of tokenized sequences (batch_size, sequence_length).

    Returns:
        torch.Tensor: A tensor of per-token losses (batch_size, sequence_length - 1).
                      Returns None if an error occurs.
    """
    if not isinstance(input_ids, torch.Tensor):
        print("Error: input_ids must be a PyTorch tensor.")
        return None
    if input_ids.ndim != 2:
        print(f"Error: input_ids must be a 2D tensor (batch_size, sequence_length), got shape {input_ids.shape}")
        return None
    if input_ids.size(1) <= 1:
        print(f"Error: sequence_length (input_ids.size(1)) must be > 1 to calculate next-token prediction loss. Got {input_ids.size(1)}")
        return None

    model.eval() # Ensure model is in evaluation mode
    print(f"Calculating per-token loss for input_ids shape: {input_ids.shape}")

    try:
        with torch.no_grad(): # No need to track gradients for loss calculation during inference/evaluation
            # Ensure input_ids are on the same device as the model
            device = next(model.parameters()).device
            input_ids_on_device = input_ids.to(device)
            
            outputs = model(input_ids=input_ids_on_device)
            logits = outputs.logits # Shape: (batch_size, sequence_length, vocab_size)
        
        if logits is None:
            print("Error: Model output logits are None.")
            return None

        # Shift logits and labels for next token prediction loss calculation
        # Logits for predicting token at position j are at logits[..., j-1, :]
        # Labels (target tokens) start from the token at position j (i.e., input_ids[..., j])
        # So, shift_logits will be logits for context 0 to N-2 (predicting tokens 1 to N-1)
        # And shift_labels will be actual tokens from 1 to N-1
        
        shift_logits = logits[..., :-1, :].contiguous()  # Shape: (batch_size, sequence_length - 1, vocab_size)
        shift_labels = input_ids_on_device[..., 1:].contiguous() # Shape: (batch_size, sequence_length - 1)

        # Calculate per-token loss using CrossEntropyLoss with reduction='none'
        loss_fct = CrossEntropyLoss(reduction='none')
        
        # Reshape for CrossEntropyLoss: 
        # shift_logits expects (N, C) where N is number of items, C is number of classes.
        # Here, N = batch_size * (sequence_length - 1), C = vocab_size.
        # shift_labels expects (N).
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape loss back to (batch_size, sequence_length - 1)
        per_token_loss = loss.view(shift_labels.shape)
        
        print(f"Successfully calculated per-token loss. Output shape: {per_token_loss.shape}")
        return per_token_loss

    except Exception as e:
        print(f"Error during per-token loss calculation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

# %%
# %% Get In-Context Loss (Full Evaluation)
import torch # Already imported, but good for cell context

def get_in_context_loss(model, test_dataloader, max_samples_to_eval: int = None):
    """
    Calculates and collects per-token loss profiles for sequences in a test dataloader,
    optionally stopping after a certain number of samples have been evaluated.

    Args:
        model: The loaded Hugging Face transformer model.
        test_dataloader: A PyTorch DataLoader yielding batches of tokenized sequences.
        max_samples_to_eval (int, optional): If set, stops evaluation once at least
                                             this many samples have been processed.
                                             Defaults to None (process all samples).

    Returns:
        list: A list of per-token loss tensors (each tensor is for one sequence).
              Returns an empty list if an error occurs or dataloader is empty.
    """
    all_loss_profiles = []
    if not test_dataloader:
        print("Error: test_dataloader is None or empty.")
        return all_loss_profiles

    model.eval() # Ensure model is in evaluation mode
    device = next(model.parameters()).device
    print(f"Starting in-context loss evaluation on device: {device}")
    
    total_batches = len(test_dataloader)
    print(f"Processing up to {total_batches} batches...")
    if max_samples_to_eval is not None:
        print(f"Evaluation will stop after approximately {max_samples_to_eval} samples.")

    evaluated_samples_count = 0

    for i, batch_input_ids in enumerate(test_dataloader):
        print(f"Processing batch {i+1}/{total_batches}, shape: {batch_input_ids.shape}")
        
        if not isinstance(batch_input_ids, torch.Tensor):
            print(f"Warning: Batch {i+1} is not a tensor, skipping. Type: {type(batch_input_ids)}")
            continue
        if batch_input_ids.ndim != 2:
            print(f"Warning: Batch {i+1} is not a 2D tensor, skipping. Shape: {batch_input_ids.shape}")
            continue

        input_ids_on_device = batch_input_ids.to(device)
        
        per_token_loss_for_batch = calculate_per_token_loss(model, input_ids_on_device)
        
        if per_token_loss_for_batch is not None:
            # Detach from graph, move to CPU, and convert to list of tensors (one per sequence)
            current_batch_profiles = [profile.cpu() for profile in per_token_loss_for_batch]
            all_loss_profiles.extend(current_batch_profiles)
            evaluated_samples_count += len(current_batch_profiles)
            print(f"Batch {i+1}: Calculated losses for {len(current_batch_profiles)} sequences. Total samples evaluated so far: {evaluated_samples_count}")

            if max_samples_to_eval is not None and evaluated_samples_count >= max_samples_to_eval:
                print(f"Reached target of {max_samples_to_eval} samples (evaluated {evaluated_samples_count}). Stopping batch processing.")
                # If we want to be super precise and not exceed max_samples_to_eval,
                # we might need to truncate all_loss_profiles here.
                if len(all_loss_profiles) > max_samples_to_eval:
                    all_loss_profiles = all_loss_profiles[:max_samples_to_eval]
                    print(f"  Truncated collected profiles to exactly {len(all_loss_profiles)}.")
                break 
        else:
            print(f"Warning: Failed to calculate per-token loss for batch {i+1}.")
            # Optionally, one could add a placeholder or skip, here we just log and continue

    print(f"In-context loss evaluation complete. Collected {len(all_loss_profiles)} loss profiles.")
    return all_loss_profiles

# %% Plotting Utility for Multiple Average Loss Profiles (Per Seed)
# import matplotlib.pyplot as plt # Already imported
# import numpy as np              # Already imported
# import os                       # Already imported
# import torch                    # Already imported

def plot_multiple_average_loss_profiles(
    seed_identifier: str,
    average_profiles_map: dict, # e.g., {"K=1": tensor_profile1, "K=5": tensor_profile2}
    output_dir: str,
    max_seq_len_to_plot: int = None,
    plot_title_prefix: str = "Average In-Context Loss Comparison"
):
    """
    Plots multiple average loss profiles (e.g., for different K-values of the same seed)
    on a single graph.

    Args:
        seed_identifier (str): Identifier for the seed (e.g., "S123"), used in title and filename.
        average_profiles_map (dict): Dictionary where keys are labels for the lines (e.g., "K=1", "K=10")
                                     and values are the 1D average loss profile tensors (or lists).
        output_dir (str): Directory to save the plot image.
        max_seq_len_to_plot (int, optional): Max sequence length to plot for all profiles.
                                             If None, uses the length of the shortest profile.
        plot_title_prefix (str): Prefix for the plot title.
    Returns:
        str: Path to the saved plot file, or None if plotting failed.
    """
    print(f"Plotting multiple average loss profiles for seed: {seed_identifier}")
    os.makedirs(output_dir, exist_ok=True)
    plot_output_file_path = None

    if not average_profiles_map:
        print("  Error: average_profiles_map is empty. Nothing to plot.")
        return None

    plt.figure(figsize=(12, 7))

    # Determine the effective maximum sequence length for plotting
    # Use the shortest profile length if max_seq_len_to_plot is not given or too long
    min_profile_len_available = float('inf')
    for label, profile_tensor in average_profiles_map.items():
        if hasattr(profile_tensor, 'shape') and len(profile_tensor.shape) > 0:
            min_profile_len_available = min(min_profile_len_available, profile_tensor.shape[0])
        elif isinstance(profile_tensor, list):
            min_profile_len_available = min(min_profile_len_available, len(profile_tensor))
    
    if min_profile_len_available == float('inf'): # No valid profiles found
        print("  Error: No valid profile lengths found in average_profiles_map.")
        plt.close() # Close the figure if created
        return None

    effective_max_len = min_profile_len_available
    if max_seq_len_to_plot is not None:
        effective_max_len = min(effective_max_len, max_seq_len_to_plot)
    
    if effective_max_len <= 0:
        print("  Error: Effective maximum sequence length to plot is zero or less.")
        plt.close()
        return None

    any_plotted = False
    for k_label, avg_loss_tensor in average_profiles_map.items():
        if hasattr(avg_loss_tensor, 'tolist'): # Convert tensor to list if it's a tensor
            avg_losses = avg_loss_tensor.tolist()
        elif isinstance(avg_loss_tensor, list):
            avg_losses = avg_loss_tensor
        else:
            print(f"  Warning: Profile for '{k_label}' is not a tensor or list, skipping.")
            continue
        
        # Trim profile to effective_max_len
        plot_losses = avg_losses[:effective_max_len]
        if not plot_losses:
            print(f"  Warning: Profile for '{k_label}' is empty after trimming to length {effective_max_len}, skipping.")
            continue

        plt.plot(range(1, len(plot_losses) + 1), plot_losses, label=str(k_label))
        any_plotted = True
        print(f"  Added profile for '{k_label}' (length: {len(plot_losses)}) to plot.")

    if not any_plotted:
        print("  Error: No data was plotted. Check profile lengths and effective_max_len.")
        plt.close()
        return None

    plt.title(f'{plot_title_prefix}\nSeed: {seed_identifier}')
    plt.xlabel('Token Position in Sequence (Context Length + 1)')
    plt.ylabel('Average Cross-Entropy Loss')
    plt.legend(title="K-value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()

    safe_seed_name_plot = seed_identifier.replace('/', '_').replace(':', '_')
    plot_file_name = f"{safe_seed_name_plot}_multi_k_avg_loss.png"
    plot_output_file_path = os.path.join(output_dir, plot_file_name)
    
    try:
        plt.savefig(plot_output_file_path)
        print(f"  Multi-K plot for seed '{seed_identifier}' saved to: {plot_output_file_path}")
    except Exception as e:
        print(f"  Error saving multi-K plot for seed '{seed_identifier}' to {plot_output_file_path}: {e}")
        plot_output_file_path = None 
    finally:
        plt.close() 
    
    return plot_output_file_path

# %% Final Analysis Main Block
if __name__ == "__main__":
    print("--- Starting Final Multi-K, Per-Seed In-Context Loss Analysis ---")

    # Configuration
    BUCKET_NAME_FINAL = 'obelisk-simplex'
    # IMPORTANT: Update this ORCHESTRATOR_PATH_FINAL to your target experiment orchestrator path
    ORCHESTRATOR_PATH_FINAL = 'non-ergodic-arxiv/training_runs/run_2025-05-11_05-33-44_final_python_orch/' 
    LOCAL_DOWNLOAD_BASE_FINAL = "analysis/downloaded_data_final"
    RESULTS_BASE_DIR_FINAL = "analysis/results_final"
    PLOTS_BASE_DIR_FINAL = "analysis/plots_final"
    MAX_EVAL_SAMPLES_PER_SEED = 1000 # Control how many samples to use for evaluation per seed
    # Ensure this is a reasonable number for your dataset size and desired speed.
    # Set to None to use all available samples in the K=1 defined dataloader.

    # Ensure output directories exist
    os.makedirs(LOCAL_DOWNLOAD_BASE_FINAL, exist_ok=True)
    os.makedirs(RESULTS_BASE_DIR_FINAL, exist_ok=True)
    os.makedirs(PLOTS_BASE_DIR_FINAL, exist_ok=True)

    # 1. List all training runs
    print(f"\\nStep 1: Listing all training runs from s3://{BUCKET_NAME_FINAL}/{ORCHESTRATOR_PATH_FINAL}")
    all_run_prefixes = list_s3_training_runs(BUCKET_NAME_FINAL, ORCHESTRATOR_PATH_FINAL)
    if not all_run_prefixes:
        print("No training runs found. Exiting.")
        # sys.exit() # Consider exiting if no runs found
    else:
        print(f"Found {len(all_run_prefixes)} total run prefixes.")

    # 2. Parse S-seed and K-values, then group runs by S-seed
    # This requires a strategy to extract S and K. Common ways:
    #   a) From the run folder name (e.g., '..._s{S}_k{K}_...')
    #   b) From the args.json file content (requires fetching metadata first)
    # Let's assume for now we can parse from folder name for simplicity,
    # but fetching metadata is more robust if names are inconsistent.
    
    # Placeholder for S-seed and K-value extraction logic
    # You'll need to adapt this regex or logic based on your actual run folder naming convention
    # Example: run_YYYY-MM-DD_HH-MM-SS_model_s{S}_k{K}_other_params/
    # We'll use a simplified regex for now. If it fails, we'll need to fetch all metadata.
    
    # Let's try to get S and K from the args_content for robustness.
    # This means we fetch metadata for ALL runs first.
    print("\\nFetching metadata for all runs to extract S and K values...")
    all_run_metadata = get_train_run_metadata(BUCKET_NAME_FINAL, all_run_prefixes)
    
    # Group runs by S-seed
    # The 'seed' or 'S' value might be in args_content.training_script_args.seed or similar.
    # The 'K' value might be in args_content.training_script_args.k_value or from the category count.
    # Adjust keys based on your _args.json structure.
    
    runs_by_seed = {}
    for meta in all_run_metadata:
        if meta and meta.get('args_content') and isinstance(meta['args_content'], dict):
            args = meta['args_content']
            # Try to find 'seed' (S value)
            s_value = args.get('seed') # V1: direct 'seed' key
            if s_value is None and 'training_script_args' in args and isinstance(args['training_script_args'], dict):
                s_value = args['training_script_args'].get('seed') # V2: nested 'seed'
            
            # Try to find 'k' (K value)
            k_value = args.get('k_value') # V1: direct 'k_value'
            if k_value is None and 'training_script_args' in args and isinstance(args['training_script_args'], dict):
                k_value = args['training_script_args'].get('k_value') # V2: nested 'k_value'
            
            # Fallback for K: count of training_categories if k_value is not explicit
            if k_value is None and meta.get('training_categories') and isinstance(meta['training_categories'], list):
                k_value = len(meta['training_categories'])

            if s_value is not None and k_value is not None:
                s_value = str(s_value) # Ensure seed is string for dict key
                k_value = str(k_value) # Ensure K is string for dict key
                
                if s_value not in runs_by_seed:
                    runs_by_seed[s_value] = []
                runs_by_seed[s_value].append({'meta': meta, 'k': k_value, 'prefix': meta['prefix']})
                print(f"  Categorized run {meta['prefix']}: S={s_value}, K={k_value}")
            else:
                print(f"  Warning: Could not determine S or K for run {meta['prefix']}. Args: {args.get('training_script_args', args)}. S attempt: {s_value}, K attempt: {k_value}")
        else:
            print(f"  Warning: Skipping run {meta.get('prefix', 'Unknown Prefix')} due to missing or invalid args_content.")

    if not runs_by_seed:
        print("Could not group any runs by S-seed. Ensure S and K values can be extracted. Exiting.")
        # sys.exit()
    else:
        print(f"\\nGrouped runs into {len(runs_by_seed)} S-seed(s): {list(runs_by_seed.keys())}")

    # 3. For each S-seed group:
    for s_seed, runs_in_seed in runs_by_seed.items():
        print(f"\\n--- Processing S-seed: {s_seed} ---")
        
        # Sort runs by K value (important for finding K=1 and processing in order)
        try:
            runs_in_seed.sort(key=lambda r: int(r['k']))
        except ValueError:
            print(f"  Warning: Could not sort runs by K for seed {s_seed} due to non-integer K values. Proceeding without K-sorting.")
        
        # 3.a Identify the K=1 run and its training categories
        k1_run_info = next((r for r in runs_in_seed if r['k'] == '1'), None)
        if not k1_run_info:
            print(f"  Error: K=1 run not found for S-seed {s_seed}. Skipping this seed.")
            continue
            
        k1_metadata = k1_run_info['meta']
        if not k1_metadata or not k1_metadata.get('training_categories'):
            print(f"  Error: K=1 run for S-seed {s_seed} (prefix: {k1_run_info.get('prefix')}) is missing training_categories in its metadata. Skipping this seed.")
            continue
        
        common_test_categories = k1_metadata['training_categories']
        print(f"  K=1 run for S-seed {s_seed} (prefix: {k1_run_info.get('prefix')}) identified.")
        print(f"  This K=1 run was trained on categories: {common_test_categories}")
        print(f"  All models for S-seed {s_seed} will be tested on these categories.")

        # 3.b Create the common test dataloader (needs a tokenizer first, get from K=1 run)
        # Download K=1 model files to get tokenizer info
        k1_s3_keys_to_download = []
        if k1_metadata.get('model_config_s3_path'):
            k1_s3_keys_to_download.append(k1_metadata['model_config_s3_path'].replace(f"s3://{BUCKET_NAME_FINAL}/", ""))
        # We don't strictly need weights for tokenizer, but config is often in same dir.
        # Let's assume config download is enough if it contains tokenizer info, or that args_content has base model.
        
        # For tokenizer, we use base_tokenizer_path from args_content of K=1 run
        k1_args_content = k1_metadata.get('args_content')
        if not k1_args_content or 'model_name_or_path' not in k1_args_content:
            print(f"  Error: 'model_name_or_path' not found in K=1 run's args_content for S-seed {s_seed}. Cannot load tokenizer. Skipping seed.")
            continue
        
        base_tokenizer_for_seed_path = k1_args_content['model_name_or_path']
        try:
            # Load a tokenizer instance just for this seed's dataloader creation
            # This tokenizer is only used for context in create_test_dataloader (RandomWindowDataset is pre-tokenized)
            # but it's good to have it for consistency and potential future use.
            seed_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_for_seed_path)
            if seed_tokenizer.pad_token is None:
                seed_tokenizer.pad_token = seed_tokenizer.eos_token
            print(f"  Loaded tokenizer for S-seed {s_seed} based on K=1 run: {base_tokenizer_for_seed_path}")
        except Exception as e_tok:
            print(f"  Error loading tokenizer {base_tokenizer_for_seed_path} for S-seed {s_seed}: {e_tok}. Skipping seed.")
            continue

        # Modify create_test_dataloader or add a wrapper to handle MAX_EVAL_SAMPLES_PER_SEED
        # For now, we'll assume create_test_dataloader's eval_multiplier can be adapted or RandomWindowDataset can be sliced.
        # A simpler approach for MAX_EVAL_SAMPLES_PER_SEED: create full dataset, then slice if needed.
        # Or, modify RandomWindowDataset to accept a max_samples parameter.
        # For now, we'll proceed and note this needs refinement if MAX_EVAL_SAMPLES_PER_SEED is strict.
        # The `eval_multiplier` in RandomWindowDataset might not directly map to `MAX_EVAL_SAMPLES_PER_SEED`.
        # Let's create a dataloader and then limit iteration if MAX_EVAL_SAMPLES is set.
        
        print(f"  Creating common test dataloader for S-seed {s_seed} with categories: {common_test_categories}")
        # Use a fixed batch size and sequence length for consistency across evaluations.
        # These should match or be appropriate for the models being tested.
        COMMON_BATCH_SIZE = 100
        COMMON_SEQUENCE_LENGTH = 100 # Adjust as needed

        # Calculate target_eval_multiplier based on MAX_EVAL_SAMPLES_PER_SEED
        # Assuming RandomWindowDataset generates roughly eval_multiplier samples per underlying paper.
        # If we want MAX_EVAL_SAMPLES_PER_SEED and we have at least 1 paper, 
        # then eval_multiplier should be MAX_EVAL_SAMPLES_PER_SEED.
        # If MAX_EVAL_SAMPLES_PER_SEED is None, we process the dataset "as is" once (eval_multiplier=1).
        if MAX_EVAL_SAMPLES_PER_SEED is not None:
            # To get up to MAX_EVAL_SAMPLES_PER_SEED, and assuming min 1 paper in category set
            target_eval_multiplier = MAX_EVAL_SAMPLES_PER_SEED 
        else:
            target_eval_multiplier = 1 # Default if not targeting a specific number of samples.
        
        print(f"  Targeting up to {MAX_EVAL_SAMPLES_PER_SEED} samples, setting eval_multiplier to: {target_eval_multiplier}")

        common_test_dataloader = create_test_dataloader(
            preprocessed_data_dir=DEFAULT_PREPROCESSED_DIR, # Assuming this global path is correct
            split_name="validation", # Or your desired test split
            tokenizer=seed_tokenizer, # For context
            batch_size=COMMON_BATCH_SIZE,
            sequence_length=COMMON_SEQUENCE_LENGTH,
            target_categories=common_test_categories,
            eval_multiplier=target_eval_multiplier, # MODIFIED: Use calculated multiplier
        )

        if not common_test_dataloader:
            print(f"  Error: Failed to create common test dataloader for S-seed {s_seed}. Skipping this seed.")
            continue
        
        print(f"  Common test dataloader for S-seed {s_seed} created. Actual batch size: {common_test_dataloader.batch_size}, Dataset size: {len(common_test_dataloader.dataset)} samples.")
        if MAX_EVAL_SAMPLES_PER_SEED is not None and len(common_test_dataloader.dataset) > MAX_EVAL_SAMPLES_PER_SEED:
            print(f"  Note: Dataloader has {len(common_test_dataloader.dataset)} samples. Will limit evaluation to approx {MAX_EVAL_SAMPLES_PER_SEED} samples if needed.")
            # Actual limiting will happen during get_in_context_loss iteration or by creating a SubsetRandomSampler/custom subset.
            # For simplicity now, get_in_context_loss will iterate, and we can stop early if needed (though it's cleaner to pass a subset).

        seed_avg_loss_profiles_map = {} # To store {"K=1": avg_profile_k1, "K=2": avg_profile_k2, ...} for this seed

        # 3.c For every run (all K-values) within this S-seed group:
        for run_info in runs_in_seed:
            current_k = run_info['k']
            current_run_prefix = run_info['prefix']
            current_run_metadata = run_info['meta']
            
            print(f"\\n    --- Processing S-seed: {s_seed}, K-value: {current_k} (Run Prefix: {current_run_prefix}) ---")
            
            # Extract original training categories for this specific K-run for verbose printing
            original_k_train_categories = current_run_metadata.get('training_categories', "Unknown/Not Found")
            print(f"      Model S{s_seed}_K{current_k} was originally trained on categories: {original_k_train_categories}")
            print(f"      Model S{s_seed}_K{current_k} is being tested on K=1 categories: {common_test_categories}")

            # Download model files for current K-run
            s3_keys_to_download_for_k = []
            model_config_key_k = current_run_metadata.get('model_config_s3_path')
            model_weights_key_k = current_run_metadata.get('model_weights_s3_path')
            args_key_k = current_run_metadata.get('args_s3_path') # For model_name_or_path

            if args_key_k:
                 s3_keys_to_download_for_k.append(args_key_k.replace(f"s3://{BUCKET_NAME_FINAL}/", ""))
            if model_config_key_k:
                s3_keys_to_download_for_k.append(model_config_key_k.replace(f"s3://{BUCKET_NAME_FINAL}/", ""))
            if model_weights_key_k:
                s3_keys_to_download_for_k.append(model_weights_key_k.replace(f"s3://{BUCKET_NAME_FINAL}/", ""))
            
            if not model_config_key_k or not model_weights_key_k or not args_key_k:
                print(f"      Error: Missing essential model/args S3 paths for S{s_seed}_K{current_k}. Skipping this model.")
                continue

            print(f"      Downloading model files for S{s_seed}_K{current_k}: {s3_keys_to_download_for_k}")
            downloaded_k_files = get_train_run_model_files(BUCKET_NAME_FINAL, s3_keys_to_download_for_k, LOCAL_DOWNLOAD_BASE_FINAL)
            
            # Determine local model directory path from downloaded files
            # This assumes config and weights are in a 'best_model' or similar subfolder.
            # We need the path to the directory containing config.json, model weights, etc.
            local_model_dir_k = None
            if model_weights_key_k: # Use weights path to find the directory
                s3_key_for_weights_k = model_weights_key_k.replace(f"s3://{BUCKET_NAME_FINAL}/", "")
                s3_dir_key_for_weights_k = os.path.dirname(s3_key_for_weights_k)
                local_model_dir_k = os.path.join(LOCAL_DOWNLOAD_BASE_FINAL, s3_dir_key_for_weights_k)
            
            if not local_model_dir_k or not os.path.exists(local_model_dir_k):
                print(f"      Error: Could not determine or find local model directory for S{s_seed}_K{current_k} at {local_model_dir_k}. Files downloaded: {downloaded_k_files}. Skipping.")
                continue

            print(f"      Local model directory for S{s_seed}_K{current_k}: {local_model_dir_k}")

            # Load model and tokenizer for current K-run
            k_run_args_content = current_run_metadata.get('args_content') # Should have been fetched already
            if not k_run_args_content: # Should not happen if metadata was fetched properly
                print(f"      Error: args_content missing for S{s_seed}_K{current_k}. Skipping.")
                continue

            # DEBUG: Print model paths from metadata before loading
            print(f"      DEBUG S{s_seed}_K{current_k}: Metadata model_source_type: {current_run_metadata.get('model_source_type')}")
            print(f"      DEBUG S{s_seed}_K{current_k}: Metadata model_config_s3_path: {current_run_metadata.get('model_config_s3_path')}")
            print(f"      DEBUG S{s_seed}_K{current_k}: Metadata model_weights_s3_path: {current_run_metadata.get('model_weights_s3_path')}")

            print(f"      Loading model and tokenizer for S{s_seed}_K{current_k}...")
            model_k, tokenizer_k = load_model_and_tokenizer(local_model_dir_k, k_run_args_content)
            if not model_k or not tokenizer_k:
                print(f"      Error: Failed to load model/tokenizer for S{s_seed}_K{current_k}. Skipping.")
                continue
            print(f"      Model and tokenizer for S{s_seed}_K{current_k} loaded successfully.")

            # Perform in-context loss calculation by manually iterating through the dataloader
            # to collect up to MAX_EVAL_SAMPLES_PER_SEED.
            # This replaces a single call to get_in_context_loss for this specific workflow.
            
            print(f"      Collecting up to {MAX_EVAL_SAMPLES_PER_SEED if MAX_EVAL_SAMPLES_PER_SEED is not None else 'all available'} samples for S{s_seed}_K{current_k} using common S-seed dataloader...")
            collected_loss_profiles_for_model_k = []
            samples_evaluated_for_model_k = 0
            
            # Max number of times to re-iterate the dataloader if it's exhausted before reaching MAX_EVAL_SAMPLES_PER_SEED.
            # This is a safeguard against very small datasets and a large MAX_EVAL_SAMPLES_PER_SEED.
            MAX_DATALOADER_REITERATIONS = 5 # Adjust as needed
            reiteration_count = 0

            # Create the initial iterator
            current_dataloader_iter = iter(common_test_dataloader)

            while True: # Loop for collecting samples, potentially across multiple dataloader passes
                if MAX_EVAL_SAMPLES_PER_SEED is not None and samples_evaluated_for_model_k >= MAX_EVAL_SAMPLES_PER_SEED:
                    print(f"      Target of {MAX_EVAL_SAMPLES_PER_SEED} samples reached for S{s_seed}_K{current_k}.")
                    break # Exit sample collection loop

                try:
                    batch_input_ids = next(current_dataloader_iter)
                    
                    if not isinstance(batch_input_ids, torch.Tensor) or batch_input_ids.ndim != 2:
                        print(f"      Warning: Skipped invalid batch from dataloader for S{s_seed}_K{current_k}.")
                        continue
                    
                    device = next(model_k.parameters()).device
                    input_ids_on_device = batch_input_ids.to(device)
                    
                    per_token_loss_for_batch = calculate_per_token_loss(model_k, input_ids_on_device)
                    
                    if per_token_loss_for_batch is not None:
                        num_in_batch = per_token_loss_for_batch.size(0)
                        
                        # Determine how many samples from this batch to add
                        samples_to_add_count = num_in_batch
                        if MAX_EVAL_SAMPLES_PER_SEED is not None:
                            needed_now = MAX_EVAL_SAMPLES_PER_SEED - samples_evaluated_for_model_k
                            samples_to_add_count = min(num_in_batch, needed_now)

                        current_batch_profiles = [profile.cpu() for profile in per_token_loss_for_batch[:samples_to_add_count]]
                        collected_loss_profiles_for_model_k.extend(current_batch_profiles)
                        samples_evaluated_for_model_k += len(current_batch_profiles)
                        
                        if samples_evaluated_for_model_k % (COMMON_BATCH_SIZE * 5) == 0 or (MAX_EVAL_SAMPLES_PER_SEED is not None and samples_evaluated_for_model_k >= MAX_EVAL_SAMPLES_PER_SEED) : 
                             print(f"        Evaluated {samples_evaluated_for_model_k}/{MAX_EVAL_SAMPLES_PER_SEED if MAX_EVAL_SAMPLES_PER_SEED is not None else 'all'} samples for S{s_seed}_K{current_k}...")
                    else:
                        print(f"      Warning: calculate_per_token_loss returned None for a batch for S{s_seed}_K{current_k}.")

                except StopIteration:
                    print(f"      Dataloader pass complete for S{s_seed}_K{current_k}. Samples collected in this pass: {samples_evaluated_for_model_k - (samples_evaluated_for_model_k % len(common_test_dataloader.dataset) if len(common_test_dataloader.dataset) > 0 else 0 ) }.") # Rough estimate of last pass
                    if MAX_EVAL_SAMPLES_PER_SEED is None: # If no max, one pass is enough.
                        print(f"      Processed all available samples ({samples_evaluated_for_model_k}) in one pass as MAX_EVAL_SAMPLES_PER_SEED is None.")
                        break # Exit sample collection loop
                    
                    if samples_evaluated_for_model_k < MAX_EVAL_SAMPLES_PER_SEED:
                        if reiteration_count < MAX_DATALOADER_REITERATIONS:
                            reiteration_count += 1
                            print(f"      Target samples ({MAX_EVAL_SAMPLES_PER_SEED}) not yet reached. Re-iterating dataloader (attempt {reiteration_count}/{MAX_DATALOADER_REITERATIONS}) for S{s_seed}_K{current_k}...")
                            current_dataloader_iter = iter(common_test_dataloader) # Get a new iterator
                        else:
                            print(f"      Warning: Reached max dataloader re-iterations ({MAX_DATALOADER_REITERATIONS}) for S{s_seed}_K{current_k}, but still have only {samples_evaluated_for_model_k}/{MAX_EVAL_SAMPLES_PER_SEED} samples. Proceeding with collected samples.")
                            break # Exit sample collection loop
                    else: # Should be caught by the check at the beginning of the while loop
                        break
                except Exception as e_batch_proc:
                    print(f"      Error processing batch for S{s_seed}_K{current_k}: {e_batch_proc}")
                    import traceback
                    traceback.print_exc()
                    break # Exit sample collection on other errors for safety

            raw_loss_profiles_k = collected_loss_profiles_for_model_k
            # The old call was:
            # raw_loss_profiles_k = get_in_context_loss(model_k, common_test_dataloader, max_samples_to_eval=MAX_EVAL_SAMPLES_PER_SEED)

            if not raw_loss_profiles_k:
                print(f"      Warning: No loss profiles collected for S{s_seed}_K{current_k}. Skipping storage and averaging.")
                continue
            
            # Store the raw loss profiles
            model_identifier_sk = f"S{s_seed}_K{current_k}"
            print(f"      Storing raw loss profiles for {model_identifier_sk}...")
            store_loss_profiles(
                {model_identifier_sk: raw_loss_profiles_k}, 
                base_output_dir=os.path.join(RESULTS_BASE_DIR_FINAL, f"S{s_seed}") # Store in seed-specific subfolder
            )

            # Calculate and collect the *average* loss profile for this (S, K) model
            # This is similar to what plot_average_loss_profile does internally before plotting.
            if raw_loss_profiles_k:
                # Ensure all profiles are lists of numbers (not tensors) for np.mean
                profiles_for_avg_k = []
                for profile in raw_loss_profiles_k:
                    if hasattr(profile, 'tolist'):
                        profiles_for_avg_k.append(profile.tolist())
                    elif isinstance(profile, list):
                        profiles_for_avg_k.append(profile)
                
                if profiles_for_avg_k:
                    try:
                        # Determine plot_len based on shortest profile or COMMON_SEQUENCE_LENGTH-1
                        min_len_profile = min(len(p) for p in profiles_for_avg_k) if profiles_for_avg_k else 0
                        avg_plot_len = min(min_len_profile, COMMON_SEQUENCE_LENGTH -1)
                        
                        valid_profiles_for_avg_k = [p[:avg_plot_len] for p in profiles_for_avg_k if len(p) >= avg_plot_len]
                        if valid_profiles_for_avg_k:
                            average_profile_k_tensor = torch.tensor(np.mean(valid_profiles_for_avg_k, axis=0)) # Store as tensor
                            seed_avg_loss_profiles_map[f"K={current_k}"] = average_profile_k_tensor # Key by "K=value" for legend
                            print(f"      Calculated and stored average loss profile for S{s_seed}_K{current_k}. Length: {len(average_profile_k_tensor)}")
                        else:
                            print(f"      Warning: No valid profiles for averaging for S{s_seed}_K{current_k} at length {avg_plot_len}.")
                    except Exception as e_avg:
                        print(f"      Error calculating average loss profile for S{s_seed}_K{current_k}: {e_avg}")
            
            # Clean up model from memory
            del model_k
            del tokenizer_k
            if 'torch' in sys.modules:
                torch.cuda.empty_cache() # If using GPU
            print(f"      Cleaned up model for S{s_seed}_K{current_k}.")


        # 4. After processing all K-values for this seed, develop and call plot_multiple_average_loss_profiles
        if seed_avg_loss_profiles_map:
            print(f"\\n  Plotting combined average loss profiles for S-seed: {s_seed}")
            # Define plot_multiple_average_loss_profiles (likely in a cell above or needs to be added to script)
            # For now, let's assume it's defined. We will define it in the next step.
            plot_multiple_average_loss_profiles(
                seed_identifier=f"S{s_seed}",
                average_profiles_map=seed_avg_loss_profiles_map, # {"K=1": tensor1, "K=2": tensor2, ...}
                output_dir=os.path.join(PLOTS_BASE_DIR_FINAL),
                max_seq_len_to_plot=COMMON_SEQUENCE_LENGTH - 1 # Or whatever is appropriate
            )
            # print(f"  (Placeholder for calling plot_multiple_average_loss_profiles for S-seed {s_seed})")
            # TODO: Implement plot_multiple_average_loss_profiles and call it here.
        else:
            print(f"  No average loss profiles collected for S-seed {s_seed} to plot.")

    print("\\n--- Final Multi-K, Per-Seed Analysis Complete ---")
