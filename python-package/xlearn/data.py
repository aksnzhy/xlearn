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

import os, ctypes
import numpy as np 
from .base import _LIB, XLearnHandle
from .base import _check_call, c_str

class DMatrix(object):
    def __init__(self, data, label=None, field_map=None):
        self.handle = ctypes.c_void_p()
        self._init_from_npy2d(data, label, field_map)
    
    def _init_from_npy2d(self, mat, label, field_map):
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')
        data = np.array(mat.reshape(mat.size), copy=False, dtype=np.float32)
        labels = np.array(label.reshape(label.size), copy=False, dtype=np.float32)
        fields = np.array(field_map.reshape(field_map.size), copy=False, dtype=np.int32)
        _check_call(_LIB.XlearnCreateDataFromMat(data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                ctypes.c_uint64(mat.shape[0]), ctypes.c_uint64(mat.shape[1]),
                                labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                fields.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                                ctypes.byref(self.handle)))
    
    def __del__(self):
        _check_call(_LIB.XlearnDataFree(ctypes.byref(self.handle)))
        self.handle = None