"""Test extracting params from dict pipeline."""

import os

import yaml

import sktune

THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_DIR, "data")


def test_extract_params():
    """Test extracting params from a pipeline YML file."""
    with open(file=os.path.join(DATA_DIR, "pipeline.yml"), encoding="utf-8") as file:
        dic = yaml.load(file, Loader=yaml.SafeLoader)
    expected = {
        "transformer__encoder__sparse": {
            "categorical": {
                "choices": [False],
            }
        },
        "regressor__learning_rate": {
            "float": {
                "low": 0.01,
                "high": 0.1,
                "log": True,
            },
        },
    }
    assert sktune.extract_params(dic)[1] == expected


def test_extract_pipeline():
    """Test extracting pipeline from a pipeline YML file."""
    with open(file=os.path.join(DATA_DIR, "pipeline.yml"), encoding="utf-8") as file:
        dic = yaml.load(file, Loader=yaml.SafeLoader)
    expected = {
        "Pipeline": {
            "steps": [
                [
                    "transformer",
                    {
                        "ColumnTransformer": {
                            "remainder": {"PowerTransformer": None},
                            "transformers": [
                                [
                                    "encoder",
                                    {
                                        "OneHotEncoder": {
                                            "handle_unknown": "ignore",
                                            "sparse": None,
                                        }
                                    },
                                    [0],
                                ],
                            ],
                        }
                    },
                ],
                [
                    "regressor",
                    {
                        "HistGradientBoostingRegressor": {
                            "learning_rate": None,
                            "loss": "squared_error",
                        }
                    },
                ],
            ]
        }
    }
    assert sktune.extract_params(dic)[0] == expected
