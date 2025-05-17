import torch
from typing import List, Union
from enum import Enum


dtype_map = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
}


class SeqProbMode(Enum):
    PROD = "prod"
    AVG = "avg"


def entropy(probabilities: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """Compute entropy for a sequence of probabilities."""
    probabilities = (
        torch.tensor(probabilities, dtype=torch.float32)
        if isinstance(probabilities, list)
        else probabilities
    )
    entropy_value = -torch.sum(probabilities * torch.log(probabilities + 1e-9))
    return entropy_value.item()
