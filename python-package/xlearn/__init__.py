# coding: utf-8

from __future__ import absolute_import
import os
from .xlearn import *

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(VERSION_FILE) as f:
	__version__ = f.read().strip()
