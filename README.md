# ArXiv Ergodic-Component Scaling Experiments

This project investigates the scaling properties of ergodic components in language models trained on subsets of the ArXiv dataset. It aims to reproduce and extend findings related to how model performance scales with the number of distinct data components (K) it's trained on.

## Project Structure

- `EXPERIMENT_PLAN.md`: Outlines the detailed experimental design, hypotheses, and evaluation metrics.
- `fetch_arxiv.py`: Script to download the `ccdv/arxiv-classification` dataset, tokenize it using a specified Pythia tokenizer, and save it in a memory-mappable format (`tokens.bin`, `index.jsonl`, `splits.json`). Supports processing the full dataset and S3 integration for download/upload.
- `random_window_dataset.py`: Defines the PyTorch `Dataset` class (`RandomWindowDataset`) for loading and sampling random 100-token windows from the preprocessed data, with support for category filtering.
- `train.py`: Script for training the language model (e.g., based on Pythia-70M architecture). It handles category selection, data loading, training loop (always starting from random model initialization, supports token budget), validation, model saving, and Weights & Biases integration.
- `requirements.txt`: Lists Python package dependencies (including `boto3` for S3, `python-dotenv`).
- `scripts/`: Contains utility shell scripts for data preparation and running experiments.
- `tests/`: Contains test scripts for various components (data preprocessing, dataset sampling, category selection).
    - `tests/outputs/`: Directory where test script outputs are saved.
- `notes.md`: Developer notes, progress tracking, and ad-hoc findings.
- `.gitignore`: Specifies intentionally untracked files.
- `preprocessed_arxiv/`: Default directory for storing processed data from `fetch_arxiv.py`. (Gitignored)
- `training_output/`: Default directory for storing training artifacts (logs, model checkpoints). (Gitignored)
- `data_cache_raw_arxiv/`: Default cache directory for the raw Hugging Face dataset. (Gitignored)
- `analysis/`: Directory containing scripts and notebooks for analyzing experiment results.
    - `analysis/analysis_script.py`: Main Python script for conducting the analysis, structured with VS Code cells for interactive execution. It will handle S3 interactions, data loading, model evaluation, and results visualization.

## Setup

This project uses [UV](https://github.com/astral-sh/uv) for fast Python environment and package management.

1.  **Install UV**:
    If you don't have UV installed, follow the official installation instructions from [astral.sh/uv](https://astral.sh/uv).

2.  **Create a Virtual Environment**:
    Navigate to the project root directory and create a virtual environment using UV:
    ```bash
    uv venv
    ```
    This will create a `.venv` directory.

3.  **Activate the Virtual Environment**:
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```
    - On Windows (PowerShell):
      ```ps1
      .venv\Scripts\Activate.ps1
      ```

4.  **Install Dependencies**:
    With the virtual environment activated, install the required packages using UV:
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **(Optional) Configure AWS Credentials for S3:**
    If you plan to use the S3 download/upload feature in `fetch_arxiv.py` or the `scripts/`:
    *   **Method 1: `.env` file (Recommended for local development):**
        Create a file named `.env` in the project root directory (this file is ignored by git). Add your AWS credentials like this:
        ```dotenv
        AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID_HERE
        AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY_HERE
        # AWS_SESSION_TOKEN=YOUR_SESSION_TOKEN_HERE # If using temporary credentials
        ```
        The `fetch_arxiv.py` script will automatically load these variables.
    *   **Method 2: Other Standard Methods:** Alternatively, ensure your credentials are configured via other standard `boto3` methods like system-wide environment variables, an IAM role (on EC2/ECS), or the `~/.aws/credentials` file.

## Running the Code

### 1. Prepare the Data

Choose one of the following methods to obtain the preprocessed data in the local `./preprocessed_arxiv/` directory:

**Method A: Process Locally**

*   **Process limited dataset (default 50 examples/split):**
    ```bash
    ./scripts/create_small_dataset_local.sh 
    # Equivalent to: python fetch_arxiv.py
    ```
*   **Process the full dataset:**
    ```bash
    python fetch_arxiv.py --full_dataset
    ```

**Method B: Use S3**

*   **First-time setup - Create and upload datasets to S3:**
    (Remember to edit S3 bucket names inside the scripts first!)
    ```bash
    chmod +x scripts/*.sh
    ./scripts/create_full_dataset_s3.sh  # Uploads full dataset to s3://<bucket>/preprocessed_arxiv/
    ./scripts/create_small_dataset_s3.sh # Uploads small dataset to s3://<bucket>/preprocessed_arxiv_small/
    ```
*   **Download a specific dataset version from S3:**
    (This skips local processing if download succeeds)
    - Download **full** dataset:
      ```bash
      python fetch_arxiv.py --download_from_s3 --s3_bucket your-bucket-name --s3_prefix preprocessed_arxiv
      ```
    - Download **small** dataset:
      ```bash
      python fetch_arxiv.py --download_from_s3 --s3_bucket your-bucket-name --s3_prefix preprocessed_arxiv_small
      ```


### 2. Run Tests (Optional but Recommended)

After preparing the data (locally or via download), run tests:
```bash
python tests/test_inspect_raw_data.py
python tests/test_inspect_processed_data.py
python tests/test_random_window_dataset_sampling.py
python tests/test_category_selection.py
```
Outputs from these tests will be saved in `tests/outputs/`.

### 3. Train the Model

The `train.py` script is used to train the language model. The model architecture is based on a specified Hugging Face model (e.g., Pythia-70M), but it is **always initialized with random weights** (i.e., trained from scratch).

You need to specify `K`, the number of categories to train on. Training duration can be controlled by `--epochs` or by a `--token_budget` (which takes precedence if set).

Example of a minimal test run (epoch-based):
```bash
python train.py --k 1 --epochs 1 --batch_size 4 --seed 42
```

Example of a run with a token budget (e.g., 1 million tokens):
```bash
python train.py --k 1 --token_budget 1000000 --batch_size 4 --seed 42
```

**Run Full Experiment Plan:**
(Assumes full dataset is available locally in `./preprocessed_arxiv/`)
```bash
chmod +x scripts/run_full_training_plan.sh
./scripts/run_full_training_plan.sh
```
(Review and adjust hyperparameters in the script if needed).

**Step-based Checkpointing:**
Add arguments like `--checkpoint_interval_steps 10000 --max_step_checkpoints 12` to the `python train.py` command or within the `run_full_training_plan.sh` script.

**Weights & Biases Integration:**
- Make sure you are logged in: `wandb login`
- Specify project/entity in the `train.py` command or within `run_full_training_plan.sh`:
  `--wandb_project "your_project_name" --wandb_entity "your_username_or_entity"`
- To disable W&B: add the `--disable_wandb` flag.

Refer to the script's arguments for more options:
```bash
python train.py --help
```

## Running on Google Colab

These instructions guide you through running the full training plan (for SEED=0) on a Google Colab instance, leveraging S3 for data storage and Weights & Biases for logging.

### 1. Setup Colab Runtime

*   Open a new Colab notebook.
*   Go to `Runtime` -> `Change runtime type`.
*   Select a GPU accelerator (e.g., T4, A100). The free tier T4 should work, but experiments will run much faster on more powerful GPUs.

### 2. Clone the Repository

Run the following cell in your Colab notebook to clone the project code:

```bash
!git clone https://github.com/Astera-org/ICL-non-ergodic-arxiv.git
%cd ICL-non-ergodic-arxiv
```

### 3. Configure Credentials (.env file - Less Secure)

**Warning:** This method is less secure than using Colab Secrets as your keys will be visible in the notebook cell. Only use this method if you understand the risks and will not share your notebook.

Create a `.env` file containing your credentials by running the following cell. Replace the placeholder values with your actual keys:

```python
%%writefile .env
WANDB_API_KEY="YOUR_WANDB_KEY_HERE"
AWS_ACCESS_KEY_ID="YOUR_AWS_ID_HERE"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_HERE"
# AWS_DEFAULT_REGION="your-region" # Optional: Uncomment and set if needed
```

Both the Python scripts (using `python-dotenv`) and the shell scripts (using the `source .env` logic we added) will load these credentials from the created `.env` file.

### 4. Install Dependencies (using uv)

We use `uv` for faster package installation. Run this cell:

```bash
!curl -LsSf https://astral.sh/uv/install.sh | sh
!source /root/.cargo/env
!uv pip install -r requirements.txt
```
*Note: If `source /root/.cargo/env` doesn't work due to shell differences in Colab, you might need to find the uv executable path (e.g., using `!which uv`) and use the full path like `!/root/.cargo/bin/uv pip install ...`*

### 5. Make Scripts Executable

Ensure the shell scripts are executable:

```bash
!chmod +x scripts/*.sh
```

### 6. Ensure Full Dataset Availability

The full training plan requires the *full* preprocessed ArXiv dataset.

*   **Prerequisite:** Ensure the *full* dataset has been generated and uploaded to S3 (e.g., by running `scripts/create_full_dataset_s3.sh` on a machine with sufficient resources, or if it already exists from a previous step).
*   **Run the download/check script:** Execute the following cell in Colab. This script will attempt to download the full dataset from the S3 location configured *within the script* (`s3://obelisk-simplex/non-ergodic-arxiv/preprocessed_arxiv` by default). If it fails (e.g., the dataset isn't there), it will attempt to process it *in Colab*, which is **not recommended** due to time and resource limits.

```bash
!bash scripts/create_full_dataset_s3.sh
```

*Verify the script output confirms the dataset was found or downloaded successfully.* If it attempts local processing, stop the cell and ensure the data exists on S3 first.

### 7. Run the Full Training Plan (Seed 0)

Now you can launch the main training script. This script will iterate through K values (1, 2, 4, 8, 11) using seed 0, log results to W&B, and upload the final outputs of each run to S3 (using the bucket/prefix configured in the script's arguments).

```bash
!bash scripts/run_full_training_plan.sh
```

### 8. Monitoring

*   **W&B:** Look for the W&B run links printed in the output of the training script. You can follow these links to monitor training progress (loss curves, etc.) in real-time.
*   **S3:** After each run finishes, check your S3 bucket under the `non-ergodic-arxiv/training_runs/` prefix (or your configured prefix) to find the output directory containing logs, checkpoints, and the final model.
*   **Colab Timeouts:** Be aware that long-running Colab sessions might time out, especially on free tiers. You may need to restart the script or use a more robust execution environment for very long training plans. Consider Colab Pro or other cloud compute options if needed.

## Contributing
(Add contribution guidelines if applicable)

## License
(Specify project license if applicable)

### Test Scripts

Several test scripts are available in the `tests/` directory:

*   `test_inspect_raw_data.py`: Loads and prints examples from the raw Hugging Face dataset.
*   `test_inspect_processed_data.py`: Loads data from `preprocessed_arxiv/`, decodes tokens, and prints samples.
*   `test_random_window_dataset_sampling.py`: Tests the window sampling and decoding from `RandomWindowDataset`.
*   `test_category_selection.py`: Tests the deterministic logic for selecting K categories.
*   `test_pretrained_in_context_loss.py`: Evaluates a pre-trained model (e.g., `EleutherAI/pythia-70m-deduped`) on the ArXiv validation set and plots its in-context learning performance (loss vs. context length). Outputs are saved to `tests/outputs/`.

To run a test script (e.g., `test_inspect_processed_data.py`): 