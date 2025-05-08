# Project Notes: ArXiv Ergodic-Component Scaling Experiments

## Current Task: Adding Full Dataset Flag and S3 Integration to Data Prep

## Plan Outline:

**Phase 1: Data Preparation**

1.  **Setup & Initial Files:**
    *   DONE: Create `notes.md` to track our progress, decisions, and any questions that arise.
    *   TODO: Create/update `README.md` with a brief project description and setup instructions as we go.
    *   DONE: Create `requirements.txt` with necessary Python packages (including `wandb`, `boto3`).
2.  **`fetch_arxiv.py`:**
    *   DONE: Download and cache the `ccdv/arxiv-classification` dataset (raw) using `no_ref` config.
    *   DONE: Tokenize text content using Pythia tokenizer.
    *   DONE: Store data as `preprocessed_arxiv/tokens.bin` (uint16 memmap), `index.jsonl`, and `splits.json`.
    *   DONE: `--max_examples_per_split` argument for limiting examples during dev/testing (default 50).
    *   DONE: `--full_dataset` flag to ignore limit and process all data.
    *   DONE: S3 Integration:
        *   `--download_from_s3` flag to attempt download first.
        *   `--upload_to_s3` flag to upload after successful local processing.
        *   `--s3_bucket` and `--s3_prefix` arguments for location.
3.  **`random_window_dataset.py` (`RandomWindowDataset` class):**
    *   DONE: Load `tokens.bin` via `np.memmap`.
    *   DONE: Load `index.jsonl` and `splits.json` to define paper pools for train/val/test.
    *   DONE: `__getitem__` samples a paper, then a random 101-token slice from it.
    *   DONE: Implement `target_categories` parameter in `__init__` to filter the paper pool for specific categories.
4.  **Test Scripts (`tests/` directory):**
    *   DONE: `test_inspect_raw_data.py` (checks raw Hugging Face dataset).
    *   DONE: `test_inspect_processed_data.py` (checks `tokens.bin`, `index.jsonl`, decodes samples).
    *   DONE: `test_random_window_dataset_sampling.py` (checks `RandomWindowDataset` output, now also serves as test for `random_window_dataset.py`'s main block).
    *   DONE: `test_category_selection.py` (tests deterministic K category selection logic).
    *   DONE: Test script outputs redirected to `tests/outputs/`.

**Phase 2: Model Training (`train.py`)**

1.  **`train.py` Script Setup:**
    *   DONE: Basic script structure: imports, argument parsing (e.g., for K, seed, epochs, batch size, learning rate, output dir for model/logs).
    *   DONE: Function to select K categories based on seed (re-use/adapt from `test_category_selection.py`).
    *   DONE: Instantiate `RandomWindowDataset` for train and validation, passing the selected categories.
    *   DONE: Instantiate `DataLoader` for train and validation sets.
    *   DONE: Weights & Biases (`wandb`) integration for logging hyperparameters and metrics.
    *   DONE: Added `--token_budget` argument to control training duration by total tokens processed.
2.  **Model Definition & Initialization (within `train.py` or imported):**
    *   DONE: Model is always initialized with random weights using its configuration (`AutoConfig` and `AutoModelForCausalLM.from_config()`).
    *   DONE: Ensure model is configured for training (e.g., `model.train()`).
3.  **Training Loop:**
    *   DONE: Optimizer (e.g., AdamW).
    *   DONE: Learning rate scheduler (configured based on total steps derived from token budget or epochs).
    *   DONE: Loop over epochs and batches, terminating early if token budget (converted to steps) is met.
    *   DONE: Forward pass, calculate loss (cross-entropy on next token prediction).
    *   DONE: Backward pass and optimizer step.
    *   DONE: Logging (loss, learning rate, maybe perplexity) - to console, log file, and `wandb`.
    *   DONE: Periodically evaluate on the validation set (results also to `wandb`).
    *   DONE: Save model checkpoints based on validation performance (best model).
    *   DONE: Added `--checkpoint_interval_steps` and `--max_step_checkpoints` for periodic step-based checkpointing.
4.  **Results and Evaluation:**
    *   DONE: Final model saving.
    *   TODO: (Later) Script to load model and evaluate perplexity on test set for chosen categories.

**Phase 3: Experiment Execution & Analysis (Defined in `EXPERIMENT_PLAN.md`)**

*   (Details to be filled in as we progress)

## Decisions & Findings:

*   Using `ccdv/arxiv-classification` with `"no_ref"` config to remove category tags from text.
*   Data stored as `preprocessed_arxiv/tokens.bin` (memmap), `index.jsonl`, and `splits.json`.
*   `RandomWindowDataset` samples 101-token windows for language modeling.
*   Category selection for training runs will be deterministic based on K and a seed.
*   `RandomWindowDataset` now supports filtering by a list of `target_categories` during initialization.
*   `train.py` implements core training loop, validation, and model saving.
*   Integrated Weights & Biases (`wandb`) into `train.py` for experiment tracking.
*   `train.py` now always initializes the model with random weights (from config) and supports step-based checkpointing via `--checkpoint_interval_steps` and `--max_step_checkpoints`.
*   `train.py` now supports a `--token_budget` argument to control training duration by the total number of tokens processed; this takes precedence over `--epochs` if set.
*   `fetch_arxiv.py` now has a `--full_dataset` flag and S3 download/upload capability.

## Questions:

*   (Empty for now) 