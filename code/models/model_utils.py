import torch
import numpy as np
import random

def set_global_seed(seed_value: int = 42) -> None:
    """
    This method sets the global seed for reproducibility

    :param seed: seed to set
    """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # For CUDA operations (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)