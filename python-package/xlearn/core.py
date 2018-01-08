# coding: utf-8

from __future__ import absolute_import

import ctypes
import numpy as np
import scipy.sparse

from .base import _LIB, _check_call, c_str
from .compat import STRING_TYPES

def c_array(ctype, values):
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)

class DMatrix(object):
    """Data Matrix used in xlearn"""

    _feature_names = None
    _field_names = None

    def __init__(self, data, label=None, field=None, silent=1,
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
            self._init_from_csr(data, field);
        elif isinstance(data, scipy.sparse.csc_matrix):
            self._init_from_csc(data, field);
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                csr_field = scipy.sparse.csr_matrix(field);
                self._init_from_csr(csr, csr_field)
            except:
                raise TypeError('can not initialize DMatrix from {}'.format(type(data).__name__))

    def _init_from_csr(self, csr, field):
        if len(csr.indices) != len(csr.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csr.indices), len(csr.data)))
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XLDMatrixCreateFromCSREx(c_array(ctypes.c_size_t, csr.indptr),
                                                  c_array(ctypes.c_uint, csr.indices),
                                                  c_array(ctypes.c_float, csr.data),
                                                  c_array(ctypes.c_uint,
                                                          field.data if field is not None else []),
                                                  ctypes.c_size_t(len(csr.indptr) - 1),
                                                  ctypes.c_size_t(len(csr.data)),
                                                  ctypes.c_size_t(csr.shape[1]),
                                                  ctypes.byref(self.handle)))

    def _init_from_csc(self, csc, field):
        if len(csc.indices) != len(csc.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csc.indices), len(csc.data)))
        self.handle = ctypes.c_void_p();
        _check_call(_LIB.XLDMatrixCreateFromCSCEx(c_array(ctypes.c_size_t, csc.indptr),
                                                  c_array(ctypes.c_uint, csc.indices),
                                                  c_array(ctypes.c_float, csc.data),
                                                  c_array(ctypes.c_uint,
                                                          field.data if field is not None else []),
                                                  ctypes.c_size_t(len(csc.indptr) - 1),
                                                  ctypes.c_size_t(len(csc.data)),
                                                  ctypes.c_size_t(csc.shape[0]),
                                                  ctypes.byref(self.handle)))

