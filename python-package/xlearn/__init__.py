# Copyright (c) 2018 by contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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