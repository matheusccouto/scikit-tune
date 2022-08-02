"""A friendly way to tune scikit-learn pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

import numpy as np
import pandas as pd
import optuna
import skdict
import sklearn
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, cross_val_score

# pylint: disable=invalid-name

FUNC_NAMES = ("categorical", "float", "int")
SKLEARN_OBJS = skdict.get_all_sklearn_objects(sklearn)


@dataclass
class Param:
    """Parameter for tuning."""

    name: str
    config: Dict[str, Dict[str, Any]]

    def evaluate(self, trial: optuna.Trial) -> float:
        """Evaluate the trial."""
        if "float" in self.config:
            return trial.suggest_float(self.name, **self.config["float"])
        if "int" in self.config:
            return trial.suggest_int(self.name, **self.config["int"])
        if "categorical" in self.config:
            return trial.suggest_categorical(self.name, **self.config["categorical"])
        raise TypeError("Unsupported param type.")


@dataclass
class Objective:
    """Optuna objective."""

    estimator: BaseEstimator
    x: pd.DataFrame | np.ndarray
    y: pd.Series | np.ndarra
    scoring: str | Callable
    cv: int | BaseCrossValidator
    params: Sequence[Param]

    def __call__(self, trial):
        self.estimator.set_params(**{p.name: p.evaluate(trial) for p in self.params})
        return cross_val_score(
            self.estimator,
            self.x,
            self.y,
            cv=self.cv,
            scoring=self.scoring,
        ).mean()


def extract_params(dic, name=None, params=None):
    """Get parameters from dict file."""
    # pylint: disable=too-many-branches
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


def tune(path, x, y, scoring, cv, n_trials, timeout, direction, output):
    """Tune scikit-learn pipelines from YAML file."""
    # pylint: disable=too-many-arguments
    with open(path, encoding="utf-8") as file:
        pipeline, params = extract_params(yaml.safe_load(file))

    estimator = skdict.load(pipeline)
    objective = Objective(
        estimator=estimator,
        x=x,
        y=y,
        scoring=scoring,
        cv=cv,
        params=[Param(key, val) for key, val in params.items()],
    )

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    estimator.set_params(**study.best_params)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, mode="w", encoding="utf-8") as file:
        yaml.safe_dump(skdict.dump(estimator), file)
