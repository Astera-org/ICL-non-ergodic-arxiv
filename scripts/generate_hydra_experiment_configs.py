import yaml
import os
from pathlib import Path
from src.experiment_config import generate_all_experiment_configs, ExperimentConfig

# Get the project root directory by going up two levels from the script's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment"

def main():
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True)

    experiment_configs = generate_all_experiment_configs()
    
    print(f"Generating {len(experiment_configs)} Hydra experiment configuration files in {CONFIG_DIR}...")

    for config in experiment_configs:
        # Sanitize active_categories for filename (replace dots, join with underscore)
        categories_str = "_".join(cat.replace(".", "") for cat in config.active_categories)
        filename = f"k{config.k_value}_{categories_str}.yaml"
        filepath = CONFIG_DIR / filename
        
        # Convert Pydantic model to dict for YAML serialization
        # Exclude ClassVars like total_token_budget_experiment if they are not meant to be in the file
        config_dict = {
            "k_value": config.k_value,
            "active_categories": config.active_categories,
            "tokens_per_category": config.tokens_per_category,
            "experiment_id": config.experiment_id # Keep experiment_id for reference
        }

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False)
        
        print(f"Generated: {filepath}")

    print("Hydra experiment configuration generation complete.")
    print(f"Make sure to add 'experiment: default_experiment_config' to your Hydra defaults list in main config.yaml.")
    print(f"You can then override it with specific experiments, e.g., `python your_app.py experiment=k1_csDS`")
    print(f"You might need to create a 'default_experiment_config.yaml' in configs/experiment/ if you want a default.")

if __name__ == "__main__":
    main() 