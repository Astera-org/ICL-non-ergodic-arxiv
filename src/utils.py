import random
import numpy as np
import torch
import os

from .logging_config import get_logger

log = get_logger(__name__)

def set_seed(seed_value: int):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # Potentially make CUDA operations deterministic, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    log.info(f"Global seed set to {seed_value}") 