"""Test tune model from YML file."""

import os
import shutil

import pytest
import skdict
from sklearn.datasets import make_regression

import sktune

THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_DIR, "data")
TMP_DIR = os.path.join(THIS_DIR, "tmp")


@pytest.fixture(name="data")
def data_fixture():
    """Generate regression data."""
    yield make_regression()


@pytest.fixture(name="clean_tmp")
def clear_tmp_fixture():
    """Generate regression data."""
    yield
    shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_tune(data, clean_tmp):  # pylint: disable=unused-argument
    """Test extracting pipeline from a pipeline YML file."""
    sktune.tune(
        path=os.path.join(DATA_DIR, "pipeline.yml"),
        x=data[0],
        y=data[1],
        scoring="r2",
        cv=5,
        n_trials=1,
        timeout=1,
        direction="maximize",
        output=os.path.join(TMP_DIR, "pipeline.yml"),
    )
    assert skdict.load(os.path.join(TMP_DIR, "pipeline.yml"))
