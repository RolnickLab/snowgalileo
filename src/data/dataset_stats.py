"""
https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469
"""

from typing import Optional

import numpy as np


class RunningStatistics:
    """
    Mean and stats over the final dimension of the data
    """

    """Records mean and variance of the final `n_dims` dimension over other dimensions across items. So collecting across `(l,m,n,o)` sized
    items with `n_dims=1` will collect `(l,m,n)` sized statistics while with `n_dims=2` the collected statistics will be of size `(l,m)`.
    Uses the algorithm from Chan, Golub, and LeVeque in "Algorithms for computing the sample variance: analysis and recommendations":
    `variance = variance1 + variance2 + n/(m*(m+n)) * pow(((m/n)*t1 - t2), 2)`
    This combines the variance for 2 blocks: block 1 having `n` elements with `variance1` and a sum of `t1` and block 2 having `m` elements
    with `variance2` and a sum of `t2`. The algorithm is proven to be numerically stable but there is a reasonable loss of accuracy (~0.1% error).
    Note that collecting minimum and maximum values is reasonably innefficient, adding about 80% to the running time, and hence is disabled by default.
    """

    def __init__(self):
        self.n: int = 0
        self.sum: Optional[np.ndarray] = None
        self._nvar: Optional[np.ndarray] = None

    def update(self, data: np.ndarray):
        if len(data.shape) > 2:
            data = np.reshape(data, [-1, data.shape[-1]])
        elif len(data.shape) == 1:
            data = np.expand_dims(data, 0)

        new_n, new_var, new_sum = data.shape[0], data.var(0), data.sum(0)
        if self.n == 0:
            self.n = new_n
            self._shape = data.shape[-1]
            self.sum = new_sum
            self._nvar = new_var * new_n
        else:
            assert (
                data.shape[-1] == self._shape
            ), f"Mismatched shapes, expected {self._shape} but got {data.shape[:-1]}."
            ratio = self.n / new_n
            t = np.power((self.sum / ratio) - new_sum, 2)
            self._nvar += new_var + ((ratio / (new_n * (self.n + new_n))) * t)
            self.sum += new_sum
            self.n += new_n

    @property
    def mean(self) -> list:
        return (self.sum / self.n).tolist() if self.n > 0 else None

    @property
    def var(self) -> np.ndarray:
        return self._nvar / self.n if self.n > 0 else None

    @property
    def std(self) -> list:
        return np.sqrt(self.var).tolist() if self.n > 0 else None
