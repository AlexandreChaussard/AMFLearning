"""Base interfaces.

Every estimator in `river` is a class, and as such inherits from at least one base interface.
These are used to categorize, organize, and standardize the many estimators that `river`
contains.

This module contains mixin classes, which are all suffixed by `Mixin`. Their purpose is to
provide additional functionality to an estimator, and thus need to be used in conjunction with a
non-mixin base class.

This module also contains utilities for type hinting and tagging estimators.

"""
from . import tags, typing
from .base import Base
from .classifier import Classifier, MiniBatchClassifier
from .estimator import Estimator

__all__ = [
    "Base",
    "Classifier",
    "Estimator",
    "MiniBatchClassifier",
]
