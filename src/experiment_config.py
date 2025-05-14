"""
Defines the Pydantic model for experiment configurations, ensuring
structure and validation for parameters defining each experimental run.
"""
from typing import List, Literal, ClassVar
from pydantic import BaseModel, validator, conint, constr
import itertools # Added for combinations

# Define the set of valid categories for our current dataset focus
VALID_CATEGORIES = Literal["cs.DS", "math.ST", "math.GR", "cs.IT", "cs.PL"]
FIVE_CATEGORIES_TUPLE = ("cs.DS", "math.ST", "math.GR", "cs.IT", "cs.PL") # For validator

# This is the reference token budget from the smallest of the top 5 categories,
# multiplied by chunk size. This is the fixed total token budget for each experiment.
# 501,760 chunks * 100 tokens/chunk = 50,176,000 tokens
REFERENCE_TOTAL_TOKEN_BUDGET = 50176000 

class ExperimentConfig(BaseModel):
    """
    Configuration for a single experimental run, defining the data sampling strategy.
    """
    experiment_id: constr(min_length=1) # type: ignore
    k_value: conint(ge=1, le=5) # Number of active categories (K)
    active_categories: List[VALID_CATEGORIES] # List of category names for this experiment
    
    # This will be constant across all experiments derived from this schema.
    # It's derived from the smallest of the top 5 categories' total chunks.
    total_token_budget_experiment: ClassVar[int] = REFERENCE_TOTAL_TOKEN_BUDGET
    
    # This is calculated based on total_token_budget_experiment and k_value
    # It represents the number of tokens to be sampled from *each* active category.
    tokens_per_category: int

    # Validators
    @validator('active_categories')
    def check_active_categories_length(cls, v, values):
        k = values.get('k_value')
        if k is not None and len(v) != k:
            raise ValueError(f"Length of active_categories ({len(v)}) must match k_value ({k}).")
        return v

    @validator('active_categories')
    def check_unique_categories(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("active_categories must contain unique category names.")
        # Ensure all categories are from the predefined valid set (implicitly handled by VALID_CATEGORIES Literal)
        # but an explicit check against a tuple/set can be more robust if Literal isn't strictly enforced everywhere.
        for cat in v:
            if cat not in FIVE_CATEGORIES_TUPLE: # Redundant with Literal but safe
                raise ValueError(f"Category '{cat}' is not one of the allowed categories: {FIVE_CATEGORIES_TUPLE}")
        return v

    @validator('tokens_per_category')
    def calculate_tokens_per_category(cls, v, values):
        # This validator acts more like a default_factory + validation if not provided,
        # or just validation if it is. We'll ensure it's correctly calculated.
        k = values.get('k_value')
        expected_tokens_per_cat = cls.total_token_budget_experiment // k if k else 0
        
        # If 'v' (the passed value for tokens_per_category) is the default int value (0)
        # it means it wasn't provided, so we calculate and assign.
        # If it was provided, we check if it's correct.
        if v == 0 and k is not None: # Assuming not provided or Pydantic default int value
             return expected_tokens_per_cat
        elif v != expected_tokens_per_cat:
            raise ValueError(
                f"tokens_per_category ({v}) is not correctly calculated. "
                f"Expected {expected_tokens_per_cat} based on total_token_budget_experiment "
                f"({cls.total_token_budget_experiment}) and k_value ({k})."
            )
        return v

    class Config:
        validate_assignment = True # Re-validate on assignment
        extra = 'forbid' # Disallow extra fields

# --- Experiment Generation ---

ALL_FIVE_CATEGORIES = FIVE_CATEGORIES_TUPLE # Alias for clarity

def generate_experiment_id(k_value: int, active_categories: List[str]) -> str:
    """Generates a somewhat readable and unique experiment ID."""
    # Sort categories to ensure consistent ID for the same set regardless of input order
    sorted_categories = sorted(active_categories)
    cat_str = "_".join(c.replace('.','') for c in sorted_categories) # e.g., csDS_mathST
    return f"K{k_value}_{cat_str}"

def generate_all_experiment_configs() -> List[ExperimentConfig]:
    """
    Generates all possible experiment configurations for K=1 through K=5
    using the ALL_FIVE_CATEGORIES set.
    """
    all_configs: List[ExperimentConfig] = []
    base_categories = list(ALL_FIVE_CATEGORIES)

    for k in range(1, len(base_categories) + 1):
        for category_combination in itertools.combinations(base_categories, k):
            active_cats = list(category_combination)
            exp_id = generate_experiment_id(k, active_cats)
            
            # tokens_per_category will be calculated by the Pydantic model
            # if we pass a default value like 0 or don't pass it and Pydantic handles it.
            # Let's rely on the validator to fill it in if not explicitly passed
            # by passing a placeholder or ensuring the model handles it.
            # The current validator requires it to be passed or defaults to 0 then validates.
            # To be safe and explicit with current validator, we pass the calculated value.
            calculated_tpc = ExperimentConfig.total_token_budget_experiment // k

            config = ExperimentConfig(
                experiment_id=exp_id,
                k_value=k,
                active_categories=active_cats, # type: ignore - Pydantic will validate Literal from list of str
                tokens_per_category=calculated_tpc 
            )
            all_configs.append(config)
    
    return all_configs


if __name__ == '__main__':
    from .logging_config import setup_logging, get_logger
    setup_logging('INFO')
    log = get_logger(__name__)

    log.info("--- Testing ExperimentConfig Pydantic Model ---")

    # Test valid configurations
    try:
        log.info("\nTesting K=1 valid config...")
        config_k1 = ExperimentConfig(
            experiment_id="K1_cs.DS",
            k_value=1,
            active_categories=["cs.DS"],
            tokens_per_category=ExperimentConfig.total_token_budget_experiment // 1 # Must be pre-calculated or handled by validator logic
        )
        assert config_k1.tokens_per_category == REFERENCE_TOTAL_TOKEN_BUDGET
        log.info(f"K=1 Valid: {config_k1.dict()}")

        log.info("\nTesting K=3 valid config...")
        config_k3 = ExperimentConfig(
            experiment_id="K3_cs.DS_math.ST_cs.IT",
            k_value=3,
            active_categories=["cs.DS", "math.ST", "cs.IT"],
            tokens_per_category=ExperimentConfig.total_token_budget_experiment // 3
        )
        assert config_k3.tokens_per_category == REFERENCE_TOTAL_TOKEN_BUDGET // 3
        log.info(f"K=3 Valid: {config_k3.dict()}")

        log.info("\nTesting K=5 valid config...")
        config_k5 = ExperimentConfig(
            experiment_id="K5_all_top_5",
            k_value=5,
            active_categories=["cs.DS", "math.ST", "math.GR", "cs.IT", "cs.PL"],
            tokens_per_category=ExperimentConfig.total_token_budget_experiment // 5
        )
        assert config_k5.tokens_per_category == REFERENCE_TOTAL_TOKEN_BUDGET // 5
        log.info(f"K=5 Valid: {config_k5.dict()}")

    except Exception as e:
        log.error(f"Error during valid config tests: {e}", exc_info=True)

    # Test invalid configurations
    invalid_test_cases = [
        ("K_mismatch", {"experiment_id": "k_mismatch", "k_value": 1, "active_categories": ["cs.DS", "cs.IT"], "tokens_per_category": 0}),
        ("duplicate_categories", {"experiment_id": "dup_cat", "k_value": 2, "active_categories": ["cs.DS", "cs.DS"], "tokens_per_category": 0}),
        ("invalid_category_name", {"experiment_id": "invalid_cat", "k_value": 1, "active_categories": ["cs.AI"], "tokens_per_category": 0}),
        ("wrong_tokens_per_cat", {"experiment_id": "wrong_tpc", "k_value": 1, "active_categories": ["cs.DS"], "tokens_per_category": 12345}),
        ("k_too_low", {"experiment_id": "k_low", "k_value": 0, "active_categories": [], "tokens_per_category": 0}),
        ("k_too_high", {"experiment_id": "k_high", "k_value": 6, "active_categories": FIVE_CATEGORIES_TUPLE + ("cs.LG",), "tokens_per_category": 0}), # cs.LG isn't valid
        ("extra_field", {"experiment_id": "extra", "k_value": 1, "active_categories": ["cs.DS"], "tokens_per_category": REFERENCE_TOTAL_TOKEN_BUDGET, "foo": "bar"}),
    ]

    for name, case in invalid_test_cases:
        log.info(f"\nTesting invalid case: {name} with data {case}")
        try:
            ExperimentConfig(**case)
            log.error(f"Validation FAILED for case '{name}'. Expected ValueError or TypeError.")
        except (ValueError, TypeError) as ve: # Pydantic can raise TypeError for wrong types too
            log.info(f"Successfully caught expected error for '{name}': {ve}")
        except Exception as e:
            log.error(f"Unexpected error for case '{name}': {e}", exc_info=True)
            
    log.info("\n--- ExperimentConfig Pydantic Model Test Complete ---")

    log.info("\n--- Generating All Experiment Configurations ---")
    all_exp_configs = generate_all_experiment_configs()
    log.info(f"Successfully generated {len(all_exp_configs)} experiment configurations.")

    # Log a few examples
    if all_exp_configs:
        log.info("\nExample generated configurations:")
        for i, cfg in enumerate(all_exp_configs):
            if i < 3 or i > len(all_exp_configs) - 4 : # Show first 3 and last 3
                log.info(f"  {cfg.experiment_id}: K={cfg.k_value}, Active={cfg.active_categories}, TPC={cfg.tokens_per_category}")
            elif i == 3:
                log.info("  ...")
    
    # Verify total count (C(5,1) + C(5,2) + C(5,3) + C(5,4) + C(5,5) = 5+10+10+5+1 = 31)
    assert len(all_exp_configs) == 31, f"Expected 31 configurations, got {len(all_exp_configs)}"
    log.info("Total number of configurations (31) matches expected count.")

    log.info("\n--- All Tests in experiment_config.py Complete ---")

"""
Python script to define and test the Pydantic model for experiment configurations.

Key features of ExperimentConfig:
- `experiment_id`: Unique string identifier.
- `k_value`: Integer (1-5), number of active categories.
- `active_categories`: List of category names, length must match `k_value`. Categories
                       must be unique and from a predefined valid set.
- `total_token_budget_experiment`: ClassVar, fixed at 50,176,000 tokens. This is the
                                   total number of tokens the model will see in an experiment.
- `tokens_per_category`: Integer, calculated as `total_token_budget_experiment // k_value`.
                         This is the number of tokens to be drawn from *each* active category.

The script includes a `__main__` block to test:
- Creation of valid ExperimentConfig instances for K=1, K=3, and K=5.
- Validation logic for various invalid cases:
    - Mismatch between `k_value` and `len(active_categories)`.
    - Duplicate categories in `active_categories`.
    - Invalid category names.
    - Incorrectly provided `tokens_per_category`.
    - `k_value` out of allowed range (0 or >5).
    - Extra fields not defined in the model.

This Pydantic model will serve as the schema for our experiment configuration files,
ensuring that each experiment run is well-defined and its parameters are consistent
with the overall experimental design.
""" 