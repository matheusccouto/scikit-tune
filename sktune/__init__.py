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

    if "Pipeline" in dic:
        for s in dic["Pipeline"]["steps"]:
            obj_name = list(s[1])[0]
            obj_values = s[1][obj_name]
            extract_params(obj_values, s[0], params)


    if obj_name in SKLEARN_OBJS:
        extract_params(obj_values, name + "__", params)

    for key in obj_values:
        for func_name in FUNC_NAMES:
            if func_name in obj_values[key]:
                params[f"{name}__{key}"] = obj_values.pop(key)
                break

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
