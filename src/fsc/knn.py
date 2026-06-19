### Original Code:
### Copyright (c) 2024 Presto Authors
### Licensed under the MIT License.
### A copy of the MIT License is available in the LICENSE file in the root directory of this project.

from sklearn.neighbors import KNeighborsClassifier as DefaultKNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor as DefaultKNeighborsRegressor


class KNNat5Classifier(DefaultKNeighborsClassifier):
    def __init__(
        self,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=5,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            weights=weights,
        )


class KNNat20Classifier(DefaultKNeighborsClassifier):
    def __init__(
        self,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=20,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            weights=weights,
        )


class KNNat100Classifier(DefaultKNeighborsClassifier):
    def __init__(
        self,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=100,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            weights=weights,
        )


class KNNat5Regressor(DefaultKNeighborsRegressor):
    def __init__(
        self,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=5,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            weights=weights,
        )


class KNNat20Regressor(DefaultKNeighborsRegressor):
    def __init__(
        self,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=20,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            weights=weights,
        )


class KNNat100Regressor(DefaultKNeighborsRegressor):
    def __init__(
        self,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=100,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            weights=weights,
        )
