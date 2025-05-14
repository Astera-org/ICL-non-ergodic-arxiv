import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import pandas as pd
import os
from transformers import AutoTokenizer

# Adjust imports based on your project structure
# Assuming analyze_dataset.py is in src/ and other modules are siblings or in PYTHONPATH
from .logging_config import setup_logging, get_logger # If logging_config is in the same directory
from .dataset_loader import ArxivDatasetLoader   # If dataset_loader is in the same directory

# If running as a script and src is the root for these modules, 
# you might need to adjust PYTHONPATH or run with `python -m src.analyze_dataset`

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def analyze_data(cfg: DictConfig) -> None:
    """Loads arXiv data, analyzes label distribution by docs and tokens, and saves visualizations."""
    setup_logging(log_level_str=cfg.get("log_level", "INFO"), log_file="analyze_dataset.log")
    log = get_logger(__name__)

    log.info("Starting dataset analysis...")
    log.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 0. Load Tokenizer
    try:
        tokenizer_name = cfg.model.tokenizer_name
        log.info(f"Loading tokenizer: {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        log.info(f"Tokenizer {tokenizer_name} loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load tokenizer {cfg.model.tokenizer_name}: {e}", exc_info=True)
        tokenizer = None # Ensure tokenizer is None if loading fails
        # Depending on requirements, you might want to exit here if tokenizer is crucial for all analyses
    
    # 1. Load the dataset
    try:
        loader = ArxivDatasetLoader(config=cfg)
        log.info("Loading training data for analysis...")
        train_dataset = loader.load_split("train")
    except Exception as e:
        log.error(f"Failed to initialize ArxivDatasetLoader or load data: {e}", exc_info=True)
        return

    if not train_dataset:
        log.error("Training data could not be loaded. Aborting analysis.")
        return
    
    log.info(f"Successfully loaded train dataset with {len(train_dataset)} samples.")

    # Get original working directory for saving plots correctly
    original_cwd = hydra.utils.get_original_cwd()
    viz_dir_base = cfg.get("results_dir", "results/visualizations")
    save_path_dir_base = os.path.join(original_cwd, viz_dir_base)
    
    if not os.path.exists(save_path_dir_base):
        os.makedirs(save_path_dir_base)
        log.info(f"Created base visualization directory: {save_path_dir_base}")

    # 2. Calculate label distribution BY DOCUMENTS
    log.info("Calculating label distribution by DOCUMENTS...")
    label_distribution_docs = loader.get_label_distribution_by_docs(train_dataset) # Corrected method name

    if not label_distribution_docs:
        log.warning("Could not retrieve document-based label distribution.")
    else:
        log.info("Label Distribution by Documents (Top 20):")
        sorted_distribution_docs = dict(sorted(label_distribution_docs.items(), key=lambda item: item[1], reverse=True))
        for i, (label, count) in enumerate(sorted_distribution_docs.items()):
            if i < 20:
                log.info(f"  {label}: {count} documents")
            else:
                break
        log.info(f"Total unique labels (by docs): {len(label_distribution_docs)}")

        # 3. Generate and save bar chart for DOCUMENT-based distribution
        log.info("Generating and saving document-based label distribution plot...")
        try:
            df_distribution_docs = pd.DataFrame(list(sorted_distribution_docs.items()), columns=['Category', 'DocCount']).head(30)
            
            plt.figure(figsize=(15, 10))
            plt.bar(df_distribution_docs['Category'], df_distribution_docs['DocCount'])
            plt.xlabel("ArXiv Category")
            plt.ylabel("Number of Documents (Train Set)")
            plt.title("Document-Based Label Distribution (Top 30 Categories)")
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            plot_filename_docs = os.path.join(save_path_dir_base, "label_distribution_docs_train_top30.png")
            plt.savefig(plot_filename_docs)
            log.info(f"Document-based label distribution plot saved to: {plot_filename_docs}")
            plt.close()

        except Exception as e:
            log.error(f"Failed to generate or save document-based plot: {e}", exc_info=True)

    # 4. Calculate label distribution BY TOKENS
    if tokenizer:
        log.info("Calculating label distribution by TOKENS...")
        # For performance on large datasets, consider sampling or running on a subset for initial analysis.
        # Here we run on the full train_dataset as per previous updates.
        label_distribution_tokens = loader.get_label_distribution_by_tokens(train_dataset, tokenizer)

        if not label_distribution_tokens:
            log.warning("Could not retrieve token-based label distribution.")
        else:
            log.info("Label Distribution by Tokens (Top 20):")
            sorted_distribution_tokens = dict(sorted(label_distribution_tokens.items(), key=lambda item: item[1], reverse=True))
            for i, (label, count) in enumerate(sorted_distribution_tokens.items()):
                if i < 20:
                    log.info(f"  {label}: {count} tokens")
                else:
                    break
            log.info(f"Total unique labels (by tokens): {len(label_distribution_tokens)}")

            # 5. Generate and save bar chart for TOKEN-based distribution
            log.info("Generating and saving token-based label distribution plot...")
            try:
                df_distribution_tokens = pd.DataFrame(list(sorted_distribution_tokens.items()), columns=['Category', 'TokenCount']).head(30)
                
                plt.figure(figsize=(15, 10))
                plt.bar(df_distribution_tokens['Category'], df_distribution_tokens['TokenCount'])
                plt.xlabel("ArXiv Category")
                plt.ylabel("Number of Tokens (Train Set)")
                plt.title("Token-Based Label Distribution (Top 30 Categories)")
                plt.xticks(rotation=90)
                plt.tight_layout()
                
                plot_filename_tokens = os.path.join(save_path_dir_base, "label_distribution_tokens_train_top30.png")
                plt.savefig(plot_filename_tokens)
                log.info(f"Token-based label distribution plot saved to: {plot_filename_tokens}")
                plt.close()

            except Exception as e:
                log.error(f"Failed to generate or save token-based plot: {e}", exc_info=True)

        # 6. Filter by top N categories (by tokens)
        top_n_val = cfg.dataset.get("num_top_categories", 5) # Corrected to use existing key
        log.info(f"Filtering dataset to keep top {top_n_val} categories based on token count...")
        
        # Ensure label_distribution_tokens is available from step 4
        if label_distribution_tokens: # Check if it was successfully computed
            # We need to pass the distribution to filter_by_top_n_categories if it's precomputed,
            # or let it recompute. The current filter_by_top_n_categories recomputes it.
            # For efficiency, if we already have `sorted_distribution_tokens`, we can extract top N from it.
            
            # The current `filter_by_top_n_categories` in `dataset_loader.py` recalculates the distribution.
            # This is acceptable. If performance becomes an issue, `dataset_loader.py` can be refactored
            # to accept a pre-calculated distribution.

            filtered_dataset_top_n_tokens = loader.filter_by_top_n_categories(
                train_dataset, 
                tokenizer=tokenizer, 
                n=top_n_val, 
                by_tokens=True
            )

            if filtered_dataset_top_n_tokens:
                log.info(f"Successfully filtered dataset to top {top_n_val} categories by tokens.")
                log.info(f"  Number of samples in filtered dataset: {len(filtered_dataset_top_n_tokens)}")
                
                # Analyze the filtered dataset's new distribution (by documents for simplicity here)
                log.info(f"  Verifying label distribution in the filtered (top {top_n_val} by tokens) dataset (document counts):")
                filtered_dist_docs_after_token_filter = loader.get_label_distribution_by_docs(filtered_dataset_top_n_tokens)
                if filtered_dist_docs_after_token_filter:
                    sorted_filtered_dist = dict(sorted(filtered_dist_docs_after_token_filter.items(), key=lambda item: item[1], reverse=True))
                    for label, count in sorted_filtered_dist.items():
                        log.info(f"    {label}: {count} documents")
                else:
                    log.warning("  Could not get document distribution for the token-filtered dataset.")
            else:
                log.error(f"Failed to filter dataset by top {top_n_val} categories (by tokens).")
        else:
            log.warning("Token-based distribution not available, skipping filtering by top N categories (by tokens).")

    else: # if tokenizer is None
        log.warning("Tokenizer was not loaded. Skipping token-based analysis and filtering.")

    log.info("Dataset analysis script finished.")

if __name__ == "__main__":
    # This allows running the script directly: `python src/analyze_dataset.py`
    # Hydra will manage the configuration loading.
    analyze_data() 