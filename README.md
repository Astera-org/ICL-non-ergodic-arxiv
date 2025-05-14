# ArXiv Ergodic-Component Scaling Experiments

This project investigates how a causal language model's in-context token-level cross entropy (XE) scales with the number (K) of statistically independent ergodic components in its training data, while holding the total number of training tokens constant.

The primary goal is to understand the trade-offs and scaling laws involved when diversifying training data across multiple distinct domains (ergodic components) versus concentrating training on fewer domains.

## Project Structure

- `data/`: Will contain raw and processed datasets.
- `src/`: Will contain source code for data processing, model training, and evaluation.
- `notebooks/`: Will contain Jupyter notebooks for exploratory data analysis and visualization.
- `experiments/`: Will contain configuration files for different experimental runs.
- `results/`: Will store the outputs and artifacts from experiments.
- `scripts/`: Contains utility scripts, including the PRD (`prd.txt`) and example PRD.

## Environment Setup

To set up the development environment, please ensure you have Python installed. It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt 
```
(Note: `requirements.txt` will be created in a subsequent task)

## Development Workflow

This project uses [Task Master AI](https://github.com/adamdotdev/task-master-ai) for managing tasks and workflow. Refer to `scripts/prd.txt` for the detailed Product Requirements Document and generated tasks.

Key commands for Task Master (ensure it's installed or use `npx task-master-ai`):

- `task-master list`: View all tasks.
- `task-master next`: Show the next recommended task.
- `task-master show <id>`: Display details for a specific task.
- `task-master set-status --id=<id> --status=<status>`: Update task status.

## Configuration and Experiment Management

This project uses [Hydra](https://hydra.cc/) for configuration management and [Weights & Biases (W&B)](https://wandb.ai/) for experiment tracking.

### Hydra Configuration

- Main configuration is in `configs/config.yaml`.
- Component configurations (dataset, model, training) are in `configs/dataset/`, `configs/model/`, and `configs/training/` respectively.
- Configurations can be overridden via the command line. For example:
  ```bash
  python src/main_hydra_app.py dataset.num_top_categories=3 training.batch_size=64
  ```
- Hydra outputs are saved to the `outputs/` directory, structured by run date and time.

### Logging

- A custom logging setup is in `src/logging_config.py`.
- Logs are written to both the console and a rotating file in the `logs/` directory (e.g., `logs/hydra_app.log`).
- Log level can be configured in `configs/config.yaml` via the `log_level` parameter.

### Experiment Tracking with Weights & Biases

- Ensure you have a W&B account and are logged in (`wandb login`).
- Set your W&B API key, project, and entity via environment variables in a `.env` file at the project root:
  ```
  WANDB_API_KEY=your_api_key
  WANDB_PROJECT=your_project_name # e.g., ICL-non-ergodic-arxiv-experiments
  WANDB_ENTITY=your_wandb_username_or_team
  ```
- Alternatively, project and entity can be set in `configs/config.yaml` under the `wandb` section.
- W&B runs will automatically log Hydra configurations, metrics, and (optionally) code.
- To disable W&B for a run, set the environment variable `WANDB_MODE=disabled` or set `wandb.mode=disabled` in the Hydra config.

### Reproducibility

- **Seeds**: The global random seed is set using the `seed` parameter in `configs/config.yaml` (see `src/utils.py`).
- **Dependencies**: Core dependencies are listed in `requirements.txt`.
- **Configuration**: Hydra configurations for each run are saved by W&B and locally in the `outputs/` directory.

## Training the Model

(Instructions for training will be added here once the model and training scripts are finalized.)

## Remote Server Setup and Dataset Generation

This project requires a tokenized dataset in HDF5 format for training. Due to its size, this file (`custom_tokenized_data_chunked_len100.hdf5`) is not tracked by Git and needs to be generated on the machine where training will occur (e.g., your remote H100 server).

Follow these steps to set up the project and generate the dataset:

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd ICL-non-ergodic-arxiv
    ```

2.  **Set up Python Environment:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
    *(On Windows, use `.venv\Scripts\activate`)*

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate the Tokenized Dataset:**
    The primary dataset file is `custom_tokenized_data_chunked_len100.hdf5`. It's generated by the `src/batch_tokenize_dataset.py` script. This script processes the raw arXiv dataset, filters it by specific categories (cs.DS, math.ST, math.GR, cs.IT, cs.PL), tokenizes the text using the custom BPE tokenizer, chunks the tokens, and saves the result to an HDF5 file.

    To generate the dataset, run the following command from the project root directory:
    ```bash
    python -m src.batch_tokenize_dataset dataset=arxiv_custom_filtered model=micro_decoder training.tokenization_batch_size=64 seed=42
    ```
    *   `dataset=arxiv_custom_filtered`: Specifies the dataset configuration, which includes paths and filtering parameters.
    *   `model=micro_decoder`: While the script mainly uses tokenizer settings from the model config, ensure this points to a config that correctly specifies your custom tokenizer (e.g., `models/custom_tokenizer/custom_bpe_tokenizer.json`). The `micro_decoder.yaml` should have `custom_tokenizer_path`.
    *   `training.tokenization_batch_size=64`: Adjust if needed based on your server's memory.
    *   `seed=42`: Ensures reproducibility.

    This script will create the `data/custom_tokenized_data_chunked_len100.hdf5` file (or a similar path if configured differently, check the script's log output for the exact location).

    **Note:** Dataset generation can take a significant amount of time and resources, depending on the size of the raw data and your machine's capabilities. The script will download the raw dataset if it's not already present in the Hugging Face cache.

5.  **Verify Dataset:**
    After the script completes, you should find the HDF5 file in the `data/` directory (or as specified in your configuration/logs). You can use the `src/analyze_tokenized_data.py` script to inspect it:
    ```bash
    python -m src.analyze_tokenized_data model=micro_decoder dataset=arxiv_custom_filtered
    ```

Once the dataset is generated, you can proceed with training the model.

(Further instructions on launching training runs will be added here.) 