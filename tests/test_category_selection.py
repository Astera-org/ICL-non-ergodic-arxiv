import random
import os
import sys
from pathlib import Path
from typing import List

# Define the 11 target categories from EXPERIMENT_PLAN.md
ALL_CATEGORIES = [
    "cs.CV", "cs.AI", "cs.SY", "cs.CE", "cs.PL", 
    "cs.IT", "cs.DS", "cs.NE", "math.AC", "math.GR", "math.ST"
]
# Ensure canonical order for reproducibility before shuffling
ALL_CATEGORIES.sort()

# Add project root for potential future imports if needed
PROJECT_ROOT = Path(__file__).parent.parent

# Output directory and file
OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "category_selection_output.txt"


def select_categories(all_categories: List[str], k: int, seed: int) -> List[str]:
    """
    Selects K categories deterministically based on a seed.

    Args:
        all_categories: The full list of available category names.
        k: The number of categories to select.
        seed: The random seed to use for shuffling.

    Returns:
        A list containing the K selected category names.
    """
    if k < 1 or k > len(all_categories):
        raise ValueError(f"K must be between 1 and {len(all_categories)}, got {k}")

    # Ensure the base list is sorted (already done globally, but good practice if passed externally)
    sorted_cats = sorted(list(all_categories)) 
    
    # Seed the random number generator
    random.seed(seed)
    
    # Shuffle a copy of the list
    shuffled_cats = list(sorted_cats) # Create a copy
    random.shuffle(shuffled_cats)
    
    # Select the first K
    selected = shuffled_cats[:k]
    
    # Return sorted selected list for consistent output comparison
    return sorted(selected) 

def run_selection_tests():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Redirect stdout to the file
    original_stdout = sys.stdout
    with open(OUTPUT_FILE, 'w') as f_out:
        sys.stdout = f_out

        print("--- Testing Category Selection ---")
        print(f"Base Categories (Sorted): {ALL_CATEGORIES}")

        k_values = [1, 2, 4, 8, 11]
        seed_values = [0, 1, 2]

        for k in k_values:
            print(f"\n--- K = {k} ---")
            for seed in seed_values:
                selected = select_categories(ALL_CATEGORIES, k=k, seed=seed)
                print(f"  Seed={seed}: Selected {len(selected)} categories: {selected}")
                # Simple assertion
                assert len(selected) == k, f"Expected {k} categories, got {len(selected)} for seed {seed}"

    # Restore original stdout
    sys.stdout = original_stdout
    print(f"Category selection test output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_selection_tests() 