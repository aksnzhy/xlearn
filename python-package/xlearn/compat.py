# coding: utf-8

from __future__ import absolute_import

import sys

PY3 = (sys.version_info[0] == 3)

if PY3:
    STRING_TYPES = str

else:
    STRING_TYPES = basestring

