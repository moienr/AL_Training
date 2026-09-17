"""Helper code for the IIASA active learning training notebooks.

The notebooks import from here so that each notebook cell stays short.
Everything is plain numpy / scikit-learn / matplotlib; the Earth Engine
part lives in earth_engine.py and is only imported by notebook 04.
"""
from . import models, strategies, simulate, toy, plotting  # noqa: F401

__version__ = "0.1"
