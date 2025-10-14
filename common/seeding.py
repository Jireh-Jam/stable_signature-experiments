import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    """Seed all common RNG sources for reproducibility.

    Args:
        seed: Seed value to use across libraries.
        deterministic_cudnn: If True, sets CuDNN to deterministic mode (may slow down).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
