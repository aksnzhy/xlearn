# coding: utf-8

from __future__ import absolute_import

import ctypes
import scipy.sparse

from .base import _LIB, _check_call, c_str
from .compat import STRING_TYPES

class DMatrix(object):
    """Data Matrix used in xlearn"""

    _feature_names = None

    def __init__(self, data, label=None, silent=None,
                 feature_names=None):
        if data is None:
            self.handle = None
            return

        if isinstance(data, STRING_TYPES):
            self.handle = ctypes.c_void_p()
            _check_call(_LIB.XLDMatrixCreateFromFile(c_str(data),
                                                     ctypes.c_int(silent),
                                                     ctypes.byref(self.handle)))
        elif isinstance(data, scipy.sparse.csr_matrix):
            self._init_from_csr(data);
        elif isinstance(data, scipy.sparse.csc_matrix):
            self._init_from_csc(data);
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self._init_from_csr(csr)
            except:
                raise TypeError('can not initialize DMatrix from {}'.format(type(data).__name__))
    
