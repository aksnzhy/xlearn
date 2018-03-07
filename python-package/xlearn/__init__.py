# coding: utf-8
from __future__ import absolute_import
import os
from .xlearn import *

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(VERSION_FILE) as f:
	__version__ = f.read().strip()

try:
	import numpy
	from sklearn.base import BaseEstimator
	from sklearn.datasets import dump_svmlight_file
	SKLEARN_INSTALLED = True

except ImportError:
	SKLEARN_INSTALLED = False

if SKLEARN_INSTALLED:
	from ._sklearn import FMModel, \
    LRModel,                       \
    FFMModel,                      \
    write_data_to_xlearn_format