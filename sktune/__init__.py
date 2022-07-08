"""A friendly way to tune scikit-learn pipelines."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd
import optuna
import sklearn
import skyaml
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, cross_val_score

# pylint: disable=invalid-name

FUNC_NAMES = ("categorical", "float", "int")
SKLEARN_OBJS = skyaml._get_all_objects(sklearn)  # pylint: disable=protected-access


class Params:
    """Parameters for tuning."""

    def evaluate(self, trial: optuna.Trial) -> float:
        """Evaluate the trial."""
        raise NotImplementedError()


@dataclass
class Objective:
    """Optuna objective."""

    estimator: BaseEstimator
    x: pd.DataFrame | np.ndarray
    y: pd.Series | np.ndarra
    scoring: str | Callable
    cv: int | BaseCrossValidator
    params: Params

    def __call__(self, trial):
        self.estimator.set_params(**self.params.evaluate(trial))
        return cross_val_score(
            self.estimator,
            self.x,
            self.y,
            cv=self.cv,
            scoring=self.scoring,
        ).mean()


def extract_params(dic, name=None, params=None):
    """Get parameters from YAML file."""
    if name is None:
        name = ""

    if params is None:
        params = {}

    for obj, val in dic.items():

        if obj in SKLEARN_OBJS:

            if obj == "Pipeline":
                steps = val["steps"]
                for new_name, objs in steps:
                    if name:
                        new_name = name + "__" + new_name
                    extract_params(objs, new_name, params)

            if obj == "ColumnTransformer":
                steps = []
                if "transformers" in val:
                    steps += val["transformers"]
                if "remainder" in val:
                    steps.append(["remainder", val["remainder"], None])
                for new_name, objs, _ in steps:
                    if name:
                        new_name = name + "__" + new_name
                    extract_params(objs, new_name, params)

            if isinstance(val, dict):
                extract_params(val, name, params)

        if isinstance(val, dict):
            for v in val:
                if v in FUNC_NAMES:
                    params[name + "__" + obj] = dic[obj]
                    dic[obj] = None

    return dic, params


# def tune(path, x, y, scoring, cv, n_trials, timeout):
#     """Tune scikit-learn pipelines from YAML file."""
#     with open(path) as file:
#         pipeline = skyaml.yaml2py(file)

#     with open(path) as file:
#         pipeline = skyaml.yaml2py(file)
#     params = pipeline.get_params()
#     estimator = pipeline.get_estimator()
#     objective = Objective(
#         estimator,
#         x,
#         y,
#         scoring,
#         cv,
#         params,
#     )
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=n_trials, timeout=timeout)
#     return study.best_params()
