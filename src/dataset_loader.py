from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
from omegaconf import DictConfig
import os
from tqdm import tqdm # Import tqdm
from collections import Counter # For efficient counting

from .logging_config import get_logger

log = get_logger(__name__)

class ArxivDatasetLoader:
    """Handles loading, Caching, and basic inspection of the arXiv dataset."""

    def __init__(self, config: DictConfig):
        """
        Initializes the dataset loader with configuration.

        Args:
            config (DictConfig): Hydra configuration object containing dataset settings.
                                 Expected keys: dataset.name, dataset.subset, dataset.cache_dir
        """
        self.dataset_name = config.dataset.name
        self.subset_name = config.dataset.subset if config.dataset.subset else None
        self.cache_dir = os.path.abspath(config.dataset.cache_dir) # Ensure absolute path for cache
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            log.info(f"Created cache directory: {self.cache_dir}")

        log.info(f"ArxivDatasetLoader initialized. Dataset: {self.dataset_name}, Subset: {self.subset_name}, Cache: {self.cache_dir}")

    def list_configs_and_splits(self):
        """Lists available configurations and splits for the dataset."""
        try:
            configs = get_dataset_config_names(self.dataset_name)
            log.info(f"Available configurations for {self.dataset_name}: {configs}")
            if self.subset_name and self.subset_name in configs:
                splits = get_dataset_split_names(self.dataset_name, config_name=self.subset_name)
                log.info(f"Available splits for {self.dataset_name} (config: {self.subset_name}): {splits}")
            elif configs:
                # Log splits for the first available config if subset_name is not specified or not found
                default_config_to_check = self.subset_name if self.subset_name else configs[0]
                if default_config_to_check not in configs: # If specified subset is not valid
                    log.warning(f"Specified subset '{default_config_to_check}' not found. Checking splits for '{configs[0]}'.")
                    default_config_to_check = configs[0]
                splits = get_dataset_split_names(self.dataset_name, config_name=default_config_to_check)
                log.info(f"Available splits for {self.dataset_name} (config: {default_config_to_check}): {splits}")
            else:
                # Fallback if no configs are found (e.g. dataset doesn't use configs)
                splits = get_dataset_split_names(self.dataset_name)
                log.info(f"Available splits for {self.dataset_name} (no specific config): {splits}")
            return configs, splits
        except Exception as e:
            log.error(f"Error listing configs/splits for {self.dataset_name}: {e}", exc_info=True)
            return [], []

    def load_split(self, split_name: str = "train"):
        """
        Loads a specific split of the dataset (e.g., "train", "validation", "test").

        Args:
            split_name (str): The name of the split to load.

        Returns:
            datasets.Dataset or None: The loaded dataset split, or None if an error occurs.
        """
        log.info(f"Attempting to load '{split_name}' split of {self.dataset_name} (subset: {self.subset_name}) from cache: {self.cache_dir}")
        try:
            dataset_split = load_dataset(
                self.dataset_name,
                name=self.subset_name, # Pass subset_name to 'name' parameter
                split=split_name,
                cache_dir=self.cache_dir
            )
            log.info(f"Successfully loaded '{split_name}' split. Features: {dataset_split.features}")
            log.info(f"Number of rows in '{split_name}' split: {len(dataset_split)}")
            # log.info(f"First example from '{split_name}' split: {dataset_split[0]}") # Be cautious with large examples
            return dataset_split
        except Exception as e:
            log.error(f"Error loading '{split_name}' split for {self.dataset_name}: {e}", exc_info=True)
            return None

    def load_all_splits(self):
        """
        Loads all available splits (train, validation, test) of the dataset.

        Returns:
            datasets.DatasetDict or None: A dictionary containing all loaded dataset splits,
                                         or None if an error occurs.
        """
        log.info(f"Attempting to load all splits of {self.dataset_name} (subset: {self.subset_name}) from cache: {self.cache_dir}")
        try:
            dataset = load_dataset(
                self.dataset_name,
                name=self.subset_name,
                cache_dir=self.cache_dir
            )
            log.info(f"Successfully loaded all splits. Available splits: {list(dataset.keys())}")
            for split_name, ds_split in dataset.items():
                log.info(f"  Split: {split_name}, Rows: {len(ds_split)}, Features: {ds_split.features}")
            return dataset
        except Exception as e:
            log.error(f"Error loading all splits for {self.dataset_name}: {e}", exc_info=True)
            return None

    def inspect_dataset(self, dataset_split, num_examples_to_show=1):
        """
        Performs and logs a basic inspection of the loaded dataset split.

        Args:
            dataset_split (datasets.Dataset): The dataset split to inspect.
            num_examples_to_show (int): Number of examples to print from the dataset.
        """
        if dataset_split is None:
            log.warning("Dataset split is None, cannot inspect.")
            return

        log.info(f"Inspecting dataset split (type: {type(dataset_split)})")
        log.info(f"  Number of rows: {len(dataset_split)}")
        log.info(f"  Features: {dataset_split.features}")
        # Example of schema: Features({'text': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=172, names=[...], id=None)})

        if len(dataset_split) > 0 and num_examples_to_show > 0:
            log.info(f"  First {min(num_examples_to_show, len(dataset_split))} example(s):")
            for i in range(min(num_examples_to_show, len(dataset_split))):
                try:
                    log.info(f"    Example {i}: {dataset_split[i]}")
                except Exception as e:
                    log.error(f"    Error accessing example {i}: {e}")
        elif len(dataset_split) == 0:
            log.warning("  Dataset split is empty.")

    def get_label_names(self, dataset_split):
        """Extracts and returns the list of label names from the dataset features."""
        if dataset_split and 'label' in dataset_split.features and hasattr(dataset_split.features['label'], 'names'):
            return dataset_split.features['label'].names
        log.warning("Could not extract label names. Dataset or label feature not as expected.")
        return []

    def get_label_distribution_by_docs(self, dataset_split):
        """Calculates and returns the distribution of labels (document counts) in a given dataset split."""
        if dataset_split is None or 'label' not in dataset_split.column_names:
            log.warning("Dataset split is None or 'label' column not found. Cannot get doc distribution.")
            return {}
        
        label_doc_counts = Counter()
        label_names_list = self.get_label_names(dataset_split)

        try:
            for example in tqdm(dataset_split, desc="Counting documents per label"):
                label_idx = example['label']
                if label_names_list and 0 <= label_idx < len(label_names_list):
                    label_name = label_names_list[label_idx]
                    label_doc_counts[label_name] += 1
                else:
                    label_doc_counts[f"unknown_label_idx_{label_idx}"] += 1 # Handle unexpected label_idx
            return dict(label_doc_counts)
        except Exception as e:
            log.error(f"Error calculating document label distribution: {e}", exc_info=True)
            return {}

    def get_label_distribution_by_tokens(self, dataset_split, tokenizer, text_column='text', label_column='label'):
        """Calculates and returns the distribution of labels by total token counts in a given dataset split."""
        if dataset_split is None or label_column not in dataset_split.column_names or text_column not in dataset_split.column_names:
            log.warning(f"Dataset split is None or '{label_column}' or '{text_column}' not found. Cannot get token distribution.")
            return {}
        if tokenizer is None:
            log.error("Tokenizer is None. Cannot calculate token distribution.")
            return {}

        label_token_counts = Counter()
        label_names_list = self.get_label_names(dataset_split)

        log.info(f"Calculating token distribution for {len(dataset_split)} documents...")
        try:
            for example in tqdm(dataset_split, desc="Counting tokens per label"):
                label_idx = example[label_column]
                text_to_tokenize = example[text_column]
                
                # Tokenize text
                # We assume tokenizer handles add_special_tokens=False and truncation=False as per EXPERIMENT_PLAN.md snippet
                # For accurate token counting for this purpose, usually no special tokens or truncation is desired.
                token_ids = tokenizer(text_to_tokenize, add_special_tokens=False, truncation=False)["input_ids"]
                num_tokens = len(token_ids)

                if label_names_list and 0 <= label_idx < len(label_names_list):
                    label_name = label_names_list[label_idx]
                    label_token_counts[label_name] += num_tokens
                else:
                    label_token_counts[f"unknown_label_idx_{label_idx}"] += num_tokens
            
            return dict(label_token_counts)
        except Exception as e:
            log.error(f"Error calculating token label distribution: {e}", exc_info=True)
            return {}

    def filter_by_specific_categories(self, dataset_split, categories_to_keep: list):
        """Filters the dataset to include only documents from the specified list of category names."""
        if not dataset_split or not categories_to_keep:
            log.warning("Dataset is None or categories_to_keep is empty. Returning original dataset.")
            return dataset_split
        
        label_names_list = self.get_label_names(dataset_split)
        if not label_names_list:
            log.error("Could not get label names from dataset. Cannot filter by specific categories.")
            return dataset_split

        # Convert category names to their corresponding indices
        indices_to_keep = set()
        for cat_name in categories_to_keep:
            try:
                indices_to_keep.add(label_names_list.index(cat_name))
            except ValueError:
                log.warning(f"Category name '{cat_name}' not found in dataset label names. It will be ignored.")

        if not indices_to_keep:
            log.warning("No valid category indices found to keep. Returning empty dataset of same type.")
            return dataset_split.filter(lambda example: False) # Return empty dataset

        log.info(f"Filtering dataset to keep categories: {categories_to_keep} (indices: {list(indices_to_keep)})")
        
        original_num_rows = len(dataset_split)
        filtered_dataset = dataset_split.filter(
            lambda example: example['label'] in indices_to_keep,
            num_proc=os.cpu_count() # Utilize multiple cores for filtering if available
        )
        log.info(f"Filtered dataset from {original_num_rows} to {len(filtered_dataset)} rows.")
        return filtered_dataset

    def filter_by_top_n_categories(self, dataset_split, tokenizer, n: int, by_tokens: bool = True):
        """Filters the dataset to include only documents from the top N categories based on document or token count."""
        if not dataset_split or n <= 0:
            log.warning("Dataset is None or n is not positive. Returning original dataset.")
            return dataset_split

        if by_tokens:
            log.info(f"Determining top {n} categories by token count...")
            distribution = self.get_label_distribution_by_tokens(dataset_split, tokenizer)
        else:
            log.info(f"Determining top {n} categories by document count...")
            distribution = self.get_label_distribution_by_docs(dataset_split)

        if not distribution:
            log.error(f"Could not get label distribution. Cannot filter by top {n} categories.")
            return dataset_split

        sorted_categories = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        
        if not sorted_categories:
            log.warning("No categories found in distribution. Returning original dataset.")
            return dataset_split
            
        top_n_category_names = [cat_name for cat_name, count in sorted_categories[:n]]
        log.info(f"Top {n} categories selected: {top_n_category_names}")

        return self.filter_by_specific_categories(dataset_split, top_n_category_names)

# Example Usage (for testing this module directly):
if __name__ == '__main__':
    from omegaconf import OmegaConf
    from .logging_config import setup_logging # Use relative import for sibling module
    from transformers import AutoTokenizer # For testing token-based distribution

    # Setup basic logging for standalone script execution
    setup_logging(log_level_str="DEBUG", log_file="dataset_loader_test.log")

    # Create a dummy Hydra config for testing
    # Make sure model.tokenizer_name is available in your actual Hydra config for the app
    dummy_config_yaml = """
    dataset:
      name: "ccdv/arxiv-classification"
      subset: "no_ref" 
      cache_dir: "./data/HF_cache_test"
    model: # Added for tokenizer testing
      tokenizer_name: "EleutherAI/pythia-70m" 
    """
    cfg = OmegaConf.create(dummy_config_yaml)

    log.info("--- Testing ArxivDatasetLoader ---")
    loader = ArxivDatasetLoader(config=cfg)
    # Initialize tokenizer for testing token-based methods
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name, use_fast=True)
    except Exception as e:
        log.error(f"Failed to load tokenizer {cfg.model.tokenizer_name} for testing: {e}")
        tokenizer = None

    log.info("--- Listing configs and splits ---")
    loader.list_configs_and_splits()

    log.info("--- Loading 'train' split ---")
    train_dataset = loader.load_split("train")
    if train_dataset:
        loader.inspect_dataset(train_dataset, num_examples_to_show=1)
        log.info("--- Getting label names ---")
        label_names = loader.get_label_names(train_dataset)
        if label_names:
            log.info(f"First 10 label names: {label_names[:10]}")
            log.info(f"Total label names: {len(label_names)}")
        
        log.info("--- Getting label distribution by DOCUMENTS for train split ---")
        train_dist_docs = loader.get_label_distribution_by_docs(train_dataset)
        if train_dist_docs:
            sorted_train_dist_docs = dict(sorted(train_dist_docs.items(), key=lambda item: item[1], reverse=True))
            log.info(f"Train split document distribution (Top 5):")
            for i, (label, count) in enumerate(sorted_train_dist_docs.items()):
                if i < 5: log.info(f"  {label}: {count}")
                else: break
        
        if tokenizer:
            log.info("--- Getting label distribution by TOKENS for train split (first 1k samples for speed) ---")
            # For a quick test, let's use a subset. Full dataset token counting can be slow.
            # In real usage, this would run on the full split needed.
            train_subset_for_token_test = train_dataset.select(range(min(1000, len(train_dataset))))
            train_dist_tokens = loader.get_label_distribution_by_tokens(train_subset_for_token_test, tokenizer)
            if train_dist_tokens:
                sorted_train_dist_tokens = dict(sorted(train_dist_tokens.items(), key=lambda item: item[1], reverse=True))
                log.info(f"Train split (1k subset) token distribution (Top 5):")
                for i, (label, count) in enumerate(sorted_train_dist_tokens.items()):
                    if i < 5: log.info(f"  {label}: {count}")
                    else: break
            
            log.info("--- Testing filter_by_top_n_categories (by tokens, n=3, on 1k subset) ---")
            filtered_top_3_tokens = loader.filter_by_top_n_categories(train_subset_for_token_test, tokenizer, n=3, by_tokens=True)
            if filtered_top_3_tokens:
                log.info(f"  Filtered dataset (top 3 by tokens) has {len(filtered_top_3_tokens)} rows.")
                loader.inspect_dataset(filtered_top_3_tokens, 1)
                filtered_dist_docs = loader.get_label_distribution_by_docs(filtered_top_3_tokens)
                log.info(f"  Distribution in filtered set (docs): {filtered_dist_docs}")

        log.info("--- Testing filter_by_specific_categories (cs.AI, math.AC, on 1k subset) ---")
        specific_cats = ['cs.AI', 'math.AC', 'cs.DS'] # cs.DS from previous run was top
        # Using train_subset_for_token_test for speed, normally you'd use train_dataset
        train_subset_for_specific_test = train_dataset.select(range(min(1000, len(train_dataset))))
        filtered_specific = loader.filter_by_specific_categories(train_subset_for_specific_test, specific_cats)
        if filtered_specific:
            log.info(f"  Filtered dataset (specific: {specific_cats}) has {len(filtered_specific)} rows.")
            loader.inspect_dataset(filtered_specific, 1)
            filtered_dist_specific_docs = loader.get_label_distribution_by_docs(filtered_specific)
            log.info(f"  Distribution in specific filtered set (docs): {filtered_dist_specific_docs}")

    log.info("--- Finished ArxivDatasetLoader Test ---") 