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

## 2024-05-08

**NaN Debugging & Training Stability:**
*   Identified and resolved a NaN issue occurring at the start of training. The root cause was traced to the AdamW optimizer's default `eps` value (`1e-8`) being too small, leading to instability on the first/second optimizer step. Changing `eps` to `1e-6` in `train.py` resolved this.
*   Experimented with `num_warmup_steps` (setting to 0, then restoring to 2000) and `learning_rate` (reduced to `2e-5`) during debugging.
*   Enhanced logging in `train.py` to include gradient norms (before/after clipping) and model parameter norms at each step for better diagnostics.
*   Updated W&B logging in `train.py` to log loss and other key metrics every step instead of every `log_interval` steps.

**Training Script Behavior:**
*   **Overfitting Observed**: User noted a U-shaped validation loss curve, indicating overfitting. The current `train.py` saves the `best_model` based on validation loss, which mitigates this.
*   **Epochs vs. Token Budget**: Clarified that `TOKEN_BUDGET` should be the primary controller of training duration. Modified `scripts/run_full_training_plan.sh` to explicitly pass `--epochs 1000` to `train.py` to ensure the token budget is met, especially for smaller `K` values where 3 epochs might finish before the token budget is exhausted.

**New Test Script:**
*   Created `tests/test_pretrained_in_context_loss.py`. This script loads a specified pre-trained Hugging Face model (e.g., `EleutherAI/pythia-70m-deduped`), evaluates it on our ArXiv validation data using the `RandomWindowDataset`, and calculates per-token losses for varying context lengths within each window. It then plots the average loss as a function of context length and saves the plot and raw data. This provides a baseline for in-context learning performance on our dataset.

**Next Steps:**
*   Run the `tests/test_pretrained_in_context_loss.py` script to get a baseline for Pythia-70M.
*   Continue monitoring the main training run with the fixes and improved logging.
*   If training is stable, consider gradually increasing the peak `LEARNING_RATE` in `scripts/run_full_training_plan.sh` towards values more typical for pre-training from scratch (e.g., starting with `1e-4` and observing, potentially going higher if stable, up to `1e-3` which was used for Pythia-70M original pre-training, keeping in mind our smaller batch size might necessitate a somewhat lower LR than their 2M token batches). 

## 2024-05-09 (Continuing NaN Debugging)

**Summary of NaN Issue Journey (leading to current state):**

1.  **Initial Problem:** `scripts/run_full_training_plan.sh` failed due to `train.py` defining `--num_workers` twice.
2.  **Fix 1:** Commented out duplicate `--num_workers` and `--force_cpu` in `train.py`.
3.  **Problem 2:** Script failed due to missing S3 upload arguments in `train.py`.
4.  **Fix 2:** Added S3 arguments (`--upload_results_to_s3`, etc.) to `train.py`.
5.  **Problem 3 (NaN Loss):** Training crashed early (micro-step 8, optimizer step 1) with NaN loss (using `bf16`), plus deprecated `torch.cuda.amp` warnings.
6.  **Fix 3 (Update Deprecated Calls):** Updated `GradScaler` and `autocast` to `torch.amp` namespace. This led to `TypeError` due to `device_type` in `GradScaler`.
7.  **Fix 4 (GradScaler TypeError):** Removed invalid `device_type` from `GradScaler`.
8.  **Problem 4 (NaN Persists + User Analysis):** NaN loss at micro-step 8 persisted. User suspected `bf16` instability or bad batch from `RandomWindowDataset`. Recommended `fp32` and logit finiteness check.
9.  **Fix 5 (Precision + Logit Check):** Changed to `fp32` in shell script. Added `outputs.logits` finiteness check in `train.py`.
10. **Problem 5 (NaN Logits):** Script failed at micro-step 8, caught by new check (`RuntimeError: Infinite/NaN logit detected`), confirming forward pass issue even with `fp32`.
11. **Debugging Step 6 (User Analysis - Bad Batch Data):** User suspected `RandomWindowDataset` still producing invalid token IDs. Proposed robust patch for `RandomWindowDataset.__getitem__`.
12. **Fix 6 (Dataset Patch):** Applied patch to `random_window_dataset.py` for better window slicing.
13. **Testing:** Created Colab snippets for forensic batch check and smoke test. After fixing `NameError: Path`, user confirmed tests passed.
14. **Problem 7 (NaN Logits Persist):** Full training script *still* failed at micro-step 8 (forward pass NaN logits).
15. **Debugging Step 7 (GPU vs CPU):** Local `debug_forward_pass.py` on CPU with problematic batch showed **finite** outputs, suggesting GPU-specific issue.
16. **Debugging Step 8 (In-Script GPU Hooks):** Modified `train.py` with detailed forward hooks and embedding checks for micro-step 8.
17. **Result 8 (Weights Corrupted):** Revealed embedding weights *themselves* contained NaN/Inf by micro-step 8, *before* the lookup for that step.
18. **Debugging Step 9 (Backward Pass Corruption):** Pointed to backward pass of micro-steps 1-7 corrupting embedding weights. Decided to use `torch.autograd.detect_anomaly`.
19. **Fix 10 (Enable detect_anomaly):** Modified `train.py` to enable `detect_anomaly`, removed redundant hooks.
20. **Unexpected Result:** Training started running without crashing. `detect_anomaly` should report, not fix.
21. **Problem 9 (Missing Logs):** WandB logs only appearing per epoch, not per optimizer step.
22. **Debugging Step 10 (Logging Logic):** Realized optimizer step logic (incl. WandB logging) was accidentally removed.
23. **Fix 11 (Restore Optimizer Logic):** Restored optimizer step, grad clipping, LR scheduling, logging, checkpointing in `train.py`.
24. **Problem 10 (NaN Loss Returns):** Script failed again at micro-step 8 (`ValueError: NaN or Inf loss detected BEFORE backward()`).
25. **Debugging Step 11 (Forward Pass NaN):** Confirmed NaN generated during forward pass/loss calculation at micro-step 8 (fp32, finite initial weights). Hypothesized subtle `autocast` issue.
26. **Fix 12 (Conditional Autocast):** Modified `train.py` to only use `torch.amp.autocast` when precision is *not* `fp32`.
27. **User Analysis (Data Loader Suspicion):** User again suspected data loader providing out-of-range indices. Proposed definitive local batch check and a more robust `RandomWindowDataset` patch.
28. **Action (Local Check):** Created and ran `temp_check_batch.py`. Confirmed saved `problematic_batch.pt` **does not contain** out-of-range token IDs.
29. **Action (Robust Dataset Patch):**
    * Applied a comprehensive patch to `random_window_dataset.py`. This patch:
        * Filters short documents upfront in `__init__` (ensuring `length >= sequence_length + 1`).
        * Adds `self.vocab_size`.
        * Introduces `_sample_window(self, paper_record)` method which:
            * Samples a window of `sequence_length + 1` tokens.
            * Performs hard checks for token ID validity (`>= 0` and `< vocab_size`) on the *entire sampled window* before returning the input part (`window[:-1]`).
            * Raises `ValueError` if bad tokens are found.
        * Modifies `__getitem__` to use `_sample_window` with a 10-attempt retry loop, selecting a new random paper on each attempt.
        * Changes `__len__` to return `len(self.pool)` (number of eligible documents).
30. **Action (Dataset Smoke Test):**
    * Created `smoke_test_dataset.py` to validate the patched `RandomWindowDataset`.
    * The script initializes the dataset, uses a `DataLoader`, and iterates through batches, performing assertions:
        * `batch.max() < dataset.vocab_size`
        * `batch.min() >= 0`
        * `torch.isfinite(batch).all()`
    * **Result:** The smoke test **passed**, indicating the patched dataset correctly samples and validates token IDs.

**Current Status:**
The `RandomWindowDataset` has been significantly hardened and a smoke test confirms it's producing valid batches locally. The puzzling NaN at micro-step 8 on Colab (GPU) persists despite `fp32`, valid initial weights, and now a seemingly robust data loader.

**Next Steps:**
*   Run the full training script (`scripts/run_full_training_plan.sh`) on the Colab GPU environment with the updated `random_window_dataset.py`.
*   Observe if the NaN issue at micro-step 8 is resolved. 

## 2024-05-10: Refactor Epoch Definition and Create Final Multi-GPU Script

**Task:** Create `run_multi_gpu_final.sh` to iterate through multiple random seeds and a fixed set of 5 `K` values. Disable ReduceLROnPlateau scheduler.

**Key Decisions/Plan:**
*   The new script `run_multi_gpu_final.sh` will be based on `run_multi_gpu_training_plan.sh`.
*   It will loop through specified random seeds.
*   For each seed, it will loop through 5 specified `K` values.
*   The `ReduceLROnPlateau` learning rate scheduler will be disabled by passing `--reduce_lr_factor 1.0` to `train.py`.
*   `MAX_EVAL_EPOCHS` in the script will remain the primary control for the number of evaluation periods.
*   `STEPS_PER_EVAL_EPOCH` will define the duration of each evaluation epoch.

**Pending Information from User:**
*   Specific 5 values of `K`.
*   List or range of random seeds. 