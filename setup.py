"""Setup Python package."""

import os
import setuptools

THIS_DIR = os.path.dirname(__file__)

with open(os.path.join(THIS_DIR, "requirements.txt"), encoding="utf-8") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="scikit-tune",
    version="0.2.2",
    author="Matheus Couto",
    author_email="matheusccouto@gmail.com",
    description="A friendly way to tune scikit-learn pipelines.",
    packages=["sktune"],
    install_requires=required,
)