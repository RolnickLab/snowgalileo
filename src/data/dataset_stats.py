"""
https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469
"""

from typing import Optional

import numpy as np


class RunningStatistics:
    """
    Mean and stats over the final dimension of the data.

    Equation 1.6 from https://apps.dtic.mil/sti/tr/pdf/ADA133112.pdf:
    """

    def __init__(self):
        self.m: int = 0
        self.sum: Optional[np.ndarray] = None
        self._nvar: Optional[np.ndarray] = None

    def update(self, data: np.ndarray):
        if len(data.shape) > 2:
            data = np.reshape(data, [-1, data.shape[-1]])
        elif len(data.shape) == 1:
            data = np.expand_dims(data, 0)

        n, new_var_div, new_sum = data.shape[0], data.var(0), data.sum(0)
        new_var = new_var_div * n
        if self.m == 0:
            self.m = n
            self._shape = data.shape[-1]
            self.sum = new_sum
            self._nvar = new_var
        else:
            assert (
                data.shape[-1] == self._shape
            ), f"Mismatched shapes, expected {self._shape} but got {data.shape[:-1]}."
            ratio = self.m / n
            t = np.power((self.sum / ratio) - new_sum, 2)
            self._nvar += new_var + ((ratio / (self.m * (self.m + n))) * t)
            self.sum += new_sum
            self.m += n

    @property
    def mean(self) -> list:
        return (self.sum / self.m).tolist() if self.m > 0 else None

    @property
    def var(self) -> np.ndarray:
        return self._nvar / self.m if self.m > 0 else None

    @property
    def std(self) -> list:
        return np.sqrt(self.var).tolist() if self.m > 0 else None
