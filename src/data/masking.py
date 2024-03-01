from collections import namedtuple
from enum import Enum
from typing import Tuple

import numpy as np


class MaskingStrategy(Enum):
    CROMA_TO_PRESTO = 0
    PRESTO_TO_CROMA = 1
    CROMA_TO_CROMA = 2
    PRESTO_TO_PRESTO = 3


MaskOutput = namedtuple(
    "MaskOutput", ["dynamic_input", "static_input", "dynamic_mask", "static_mask"]
)


def subset_image(
    dynamic_input: np.ndarray, static_input: np.ndarray, size: int
) -> Tuple[np.ndarray, np.ndarray]:
    assert (dynamic_input.shape[0] == static_input.shape[0]) & (
        dynamic_input.shape[1] == static_input.shape[1]
    )
    possible_h = dynamic_input.shape[0] - size
    possible_w = dynamic_input.shape[1] - size
    assert (possible_h >= 0) & (possible_w >= 0)

    if possible_h > 0:
        start_h = np.random.choice(possible_h)
    else:
        start_h = possible_h

    if possible_w > 0:
        start_w = np.random.choice(possible_w)
    else:
        start_w = possible_w

    return dynamic_input[start_h : start_h + size, start_w : start_w + size], static_input[
        start_h : start_h + size, start_w : start_w + size
    ]
