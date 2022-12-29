"""Datasets.

This module contains a collection of datasets for multiple tasks: classification, regression, etc.
The data corresponds to popular datasets and are conveniently wrapped to easily iterate over
the data in a stream fashion. All datasets have fixed size. Please refer to `river.synth` if you
are interested in infinite synthetic data generators.

"""
from . import base, synth
from .bananas import Bananas

__all__ = [
    "Bananas",
    "base",
]
