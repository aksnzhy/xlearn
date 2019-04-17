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
import sys
import os
import ctypes
from numpy import ndarray
from .base import Series, DataFrame 
import numpy as np 
from .base import _LIB, XLearnHandle
from .base import _check_call, c_str
from .data import DMatrix

class XLearn(object):
    """XLearn is the core interface used by python API."""

    def __init__(self, handle):
        """Initalizes a new XLearn

        Parameters
        ----------
        handle : XLearnHandle
            'XLearn' handle of C API.
        """
        assert isinstance(handle, XLearnHandle)
        self.handle = handle

    def __del__(self):
        _check_call(_LIB.XLearnHandleFree(ctypes.byref(self.handle)))

    def _set_Param(self, param):
        """Set hyper-parameter for xlearn handle

        Parameters
        ----------
        param : dict
            xlearn hyper-parameters
        """
        for (key, value) in param.items():
            if key == 'task':
                _check_call(_LIB.XLearnSetStr(ctypes.byref(self.handle),
                                              c_str(key), c_str(value)))
            elif key == 'metric':
                _check_call(_LIB.XLearnSetStr(ctypes.byref(self.handle),
                                              c_str(key), c_str(value)))
            elif key == 'opt':
                _check_call(_LIB.XLearnSetStr(ctypes.byref(self.handle),
                                              c_str(key), c_str(value)))
            elif key == 'log':
                _check_call(_LIB.XLearnSetStr(ctypes.byref(self.handle),
                                              c_str(key), c_str(value)))
            elif key == 'lr':
                _check_call(_LIB.XLearnSetFloat(ctypes.byref(self.handle),
                                                c_str(key), ctypes.c_float(value)))
            elif key == 'k':
                _check_call(_LIB.XLearnSetInt(ctypes.byref(self.handle),
                                              c_str(key), ctypes.c_uint(value)))
            elif key == 'lambda':
                _check_call(_LIB.XLearnSetFloat(ctypes.byref(self.handle),
                                                c_str(key), ctypes.c_float(value)))
            elif key == 'init':
                _check_call(_LIB.XLearnSetFloat(ctypes.byref(self.handle),
                                                c_str(key), ctypes.c_float(value)))
            elif key == 'epoch':
                _check_call(_LIB.XLearnSetInt(ctypes.byref(self.handle),
                                              c_str(key), ctypes.c_uint(value)))
            elif key == 'fold':
                _check_call(_LIB.XLearnSetInt(ctypes.byref(self.handle),
                                              c_str(key), ctypes.c_uint(value)))
            elif key == 'alpha':
                _check_call(_LIB.XLearnSetFloat(ctypes.byref(self.handle),
                                                c_str(key), ctypes.c_float(value)))
            elif key == 'beta':
                _check_call(_LIB.XLearnSetFloat(ctypes.byref(self.handle),
                                                c_str(key), ctypes.c_float(value)))
            elif key == 'lambda_1':
                _check_call(_LIB.XLearnSetFloat(ctypes.byref(self.handle),
                                                c_str(key), ctypes.c_float(value)))
            elif key == 'lambda_2':
                _check_call(_LIB.XLearnSetFloat(ctypes.byref(self.handle),
                                                c_str(key), ctypes.c_float(value)))
            elif key == 'nthread':
                _check_call(_LIB.XLearnSetInt(ctypes.byref(self.handle),
                                              c_str(key), ctypes.c_uint(value)))
            elif key == 'block_size':
                _check_call(_LIB.XLearnSetInt(ctypes.byref(self.handle),
                                              c_str(key), ctypes.c_uint(value)))
            elif key == 'stop_window':
                _check_call(_LIB.XLearnSetInt(ctypes.byref(self.handle),
                                              c_str(key), ctypes.c_uint(value)))
            elif key == 'seed':
                _check_call(_LIB.XLearnSetInt(ctypes.byref(self.handle),
                                              c_str(key), ctypes.c_uint(value)))
            else:
                raise Exception("Invalid key!", key)

    def show(self):
        """Show model information
        """
        _check_call(_LIB.XLearnShow(ctypes.byref(self.handle)))

    def setTrain(self, train_path):
        """Set file path of training data.

        Parameters
        ----------
        train_path : str
           the path of training data
        """
        if isinstance(train_path, str):
            _check_call(_LIB.XLearnSetTrain(ctypes.byref(self.handle), c_str(train_path)))
            _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle), c_str("from_file"), ctypes.c_bool(True)))
        elif isinstance(train_path, DMatrix):
            key = "train"
            _check_call(_LIB.XLearnSetDMatrix(ctypes.byref(self.handle), c_str(key), ctypes.byref(train_path.handle)))
            _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle), c_str("from_file"), ctypes.c_bool(False)))
        else:
            raise Exception("Invalid train.Can be test file path or xLearn DMatrix", type(train_path))

    def setTest(self, test_path):
        """Set file path of test data.

        Parameters
        ----------
        test_path : str
           the path of test data.
        """
        if isinstance(test_path, str):
            _check_call(_LIB.XLearnSetTest(ctypes.byref(self.handle), c_str(test_path)))
            _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle), c_str("from_file"), ctypes.c_bool(True)))
        elif isinstance(test_path, DMatrix):
            key = "test"
            _check_call(_LIB.XLearnSetDMatrix(ctypes.byref(self.handle), c_str(key), ctypes.byref(test_path.handle)))
            _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle), c_str("from_file"), ctypes.c_bool(False)))
        else:
            raise Exception("Invalid test.Can be test file path or xLearn DMatrix", type(test_path))

    def setPreModel(self, pre_model_path):
        """ Set file path of pre-trained model.

        Parameters
        ----------
        pre_model_path : str
           the path of pre-trained model.
        """
        _check_call(_LIB.XLearnSetPreModel(ctypes.byref(self.handle), c_str(pre_model_path)))

    def setValidate(self, val_path):
        """Set file path of validation data.

        Parameters
        ----------
        val_path : str
           the path of validation data.
        """
        if isinstance(val_path, str):
            _check_call(_LIB.XLearnSetValidate(ctypes.byref(self.handle), c_str(val_path)))
        elif isinstance(val_path, DMatrix):
            key = "validate"
            _check_call(_LIB.XLearnSetDMatrix(ctypes.byref(self.handle), c_str(key), ctypes.byref(val_path.handle)))
        else:
            raise Exception("Invalid validation.Can be test file path or xLearn DMatrix", type(val_path))

    def setTXTModel(self, model_path):
        """Set the path of TXT model file.

        Parameters
        ----------
        model_path : str
            the path of the TXT model file.
        """
        _check_call(_LIB.XLearnSetTXTModel(ctypes.byref(self.handle), c_str(model_path)))

    def setQuiet(self):
        """Set xlearn to quiet model"""
        key = 'quiet'
        _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle),
                                       c_str(key), ctypes.c_bool(True)))

    def setOnDisk(self):
        """Set xlearn to use on-disk training"""
        key = 'on_disk'
        _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle),
                                       c_str(key), ctypes.c_bool(True)))

    def setNoBin(self):
        """Do not generate bin file"""
        key = 'bin_out'
        _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle),
                                       c_str(key), ctypes.c_bool(False)))

    def disableNorm(self):
        """Disable instance-wise normalization"""
        key = 'norm'
        _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle),
                                       c_str(key), ctypes.c_bool(False)))

    def disableLockFree(self):
        """Disable lock free training"""
        key = 'lock_free'
        _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle),
                                       c_str(key), ctypes.c_bool(False)))

    def disableEarlyStop(self):
        """Disable early-stopping"""
        key = 'early_stop'
        _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle),
                                       c_str(key), ctypes.c_bool(False)))

    def setSign(self):
        """Convert output to 0 and 1"""
        key = 'sign'
        _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle),
                                       c_str(key), ctypes.c_bool(True)))

    def setSigmoid(self):
        """Convert output by using sigmoid"""
        key = 'sigmoid'
        _check_call(_LIB.XLearnSetBool(ctypes.byref(self.handle),
                                       c_str(key), ctypes.c_bool(True)))

    def fit(self, param, model_path):
        """Check hyper-parameters, train model, and dump model.

        Parameters
        ----------
        param : dict
          hyper-parameter used by xlearn.
        model_path : str
          path of model checkpoint.
        """
        self._set_Param(param)
        _check_call(_LIB.XLearnFit(ctypes.byref(self.handle), c_str(model_path)))

    def cv(self, param):
        """ Do cross-validation

        Parameters
        ----------
        param : dict
          hyper-parameter used by xlearn
        """
        self._set_Param(param)
        _check_call(_LIB.XLearnCV(ctypes.byref(self.handle)))

    def predict(self, model_path, out_path=None):
        """Predict output

        Parameters
        ----------
        model_path : str. path of model checkpoint.
        out_path : str, default None. if a path of output result is setted, then will save result to local file,
        and will not return numpy res.
        """
        if out_path is None:
            length = ctypes.c_uint64()
            preds = ctypes.POINTER(ctypes.c_float)()
            _check_call(_LIB.XLearnPredictForMat(ctypes.byref(self.handle),
                                                 c_str(model_path),
                                                 ctypes.byref(length),
                                                 ctypes.byref(preds)))
            res = np.zeros(length.value, dtype=np.float32)
            ctypes.memmove(res.ctypes.data, preds, length.value * res.strides[0])
            return res
        else:
            _check_call(_LIB.XLearnPredictForFile(ctypes.byref(self.handle),
                                                 c_str(model_path),
                                                 c_str(out_path)))

def create_linear():
    """
    Create a linear model.
    """
    model_type = 'linear'
    handle = XLearnHandle()
    _check_call(_LIB.XLearnCreate(c_str(model_type), ctypes.byref(handle)))
    return XLearn(handle)


def create_fm():
    """
    Create a factorization machine.
    """
    model_type = 'fm'
    handle = XLearnHandle()
    _check_call(_LIB.XLearnCreate(c_str(model_type), ctypes.byref(handle)))
    return XLearn(handle)


def create_ffm():
    """
    Create a field-aware factorization machine.
    """
    model_type = 'ffm'
    handle = XLearnHandle()
    _check_call(_LIB.XLearnCreate(c_str(model_type), ctypes.byref(handle)))
    return XLearn(handle)


def hello():
    """
    Say hello to user
    """
    _check_call(_LIB.XLearnHello())
