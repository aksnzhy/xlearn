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
from numpy import ndarray
from .base import DataFrame, Series  
import numpy as np 
from .base import _LIB, XLearnHandle
from .base import _check_call, c_str

# This class is the xLearn core data
class DMatrix(object):
    def __init__(self, data, label=None, field_map=None):
        """
        Initial function.
        Parameters:
        data: NumPy 2D or pandas DataFrame of features data.
        label: one-dimentional array, it presents samples label.
        field_map: one-dimentional array, it presents the feautres'field repsepctively.
        This field_map like, [1, 2, 1, 3] means, the first and third features belong to field one, and the second belongs to field two, and so on.
        this parameter only useful for ffm model.
        Note: we only do roughly check, and do detail check in true work function.
        """

        self.__handle = ctypes.c_void_p()
        if (isinstance(data, ndarray) or isinstance(data, DataFrame)):
            self._init_from_npy2d(data, label, field_map)
        else:
            raise ValueError('Input data must be numpy.ndarray or pandas.DataFrame')
    
    # TODO(etveritas): support init DMatrix from other data type in memory, 
    # and unite init DMatrix from file.

    # we now, only define the function for init DMatrix from NumPy 2D and pandas DataFrame.
    def _init_from_npy2d(self, mat, label, field_map):
        """
        This function do initialize DMatrix from numpy 2D and pandas DataFrame.
        Parameters
        """
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')

        if isinstance(mat, DataFrame):
            mat = mat.values

        data = np.array(mat.reshape(mat.size), copy=False, dtype=np.float32)
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        label_ptr = None
        field_ptr = None
        if label is not None:
            if isinstance(label, DataFrame):
                label = label.values
            if isinstance(label, Series):
                label = label.values
            if isinstance(label, list):
                label = np.array(label)
            if isinstance(label, ndarray):
                if (len(label.shape) > 2):
                    raise ValueError('Input numpy.ndarray of label must be 1 dimensional or 2 dimensional with one dimensional is 1')
                if (len(label.shape) == 2) and (label.shape[0] != 1) and (label.shape[1] != 1):
                    print(len(label.shape))
                    raise ValueError('Input numpy.ndarray of label must be 1 dimensional or 2 dimensional with one dimensional is 1')
                if (label.size != mat.shape[0]):
                    raise ValueError('Input label must has same elements as the data lines')
                labels = np.array(label.reshape(label.size), copy=False, dtype=np.float32)
                label_ptr = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            else:
                raise ValueError('Input label must be numpy.ndarray')

        if field_map is not None:
            if isinstance(field_map, DataFrame):
                field_map = field_map.values
            if isinstance(field_map, Series):
                field_map = field_map.values
            if isinstance(field_map, list):
                field_map = np.array(field_map)
            if isinstance(field_map, ndarray):
                if (len(field_map.shape) > 2):
                    raise ValueError('Input numpy.ndarray of label must be 1 dimensional or 2 dimensional with one dimensional is 1')
                if (len(field_map.shape) == 2) and (field_map.shape[0] != 1) and (label.shape[1] != 1):
                    raise ValueError('Input numpy.ndarray of field_map must be 1 dimensional or 2 dimensional with the one dimensional is 1')
                if (field_map.size != mat.shape[1]):
                    raise ValueError('Input field_map must has same elements as the data columns')
                fields = np.array(field_map.reshape(field_map.size), copy=False, dtype=np.int32)
                field_ptr = fields.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
            else:
                raise ValueError('Input of field_map must numpy.ndarray')

        _check_call(_LIB.XlearnCreateDataFromMat(data_ptr,
                                                 ctypes.c_uint64(mat.shape[0]),
                                                 ctypes.c_uint64(mat.shape[1]),
                                                 label_ptr,
                                                 field_ptr,
                                                 ctypes.byref(self.__handle)))

    @property
    def handle(self):
        return self.__handle

    # release the resource of DMatrix
    def __del__(self):
        _check_call(_LIB.XlearnDataFree(ctypes.byref(self.__handle)))
        self.__handle = None