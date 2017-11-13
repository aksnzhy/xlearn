# coding: utf-8
import sys
import os
import ctypes

class XLearnError(Exception):
	"""Error thrown by xlearn trainer"""
    pass

def _load_lib():
	"""Load xlearn library"""
	lib_path = find_lib_path()
	if len(lib_path) == 0:
		return None
	lib = ctypes.cdll.Loadlibrary(lib_path[0])
	return lib

# load the xlearn library globally
_LIB = _load_lib()

def _check_call(ret):
	"""Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
	"""
	if ret != 0:
		raise XLearnError(_LIB.XLearGetLastError())

#type definitions
XLearnHandle = ctypes.c_void_p

class XLearn(object):
	"""XLearn the core interface used by python API."""
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
		_check_call(_LIB.XLearnHandleFree(self.handle))

	def setTrain(self, train_path):
		"""Set file path of training data.

		Parameters
		----------
		train_path : str
		   the path of training data
		"""
		_check_call(_LIB.XLearnSetTrain(self.handle, c_str(train_path)))

	def setTest(self, test_path):
		"""Set file path of test data.

		Parameters
		----------
		test_path : str
		   the path of test data.
		"""
		_check_call(_LIB.XLearnSetTest(self.handle, c_str(test_path)))

	def setValidate(self, val_path):
		"""Set file path of validation data.

		Parameters
		----------
		val_path : str
		   the path of validation data.
		"""
		_check_call(_LIB.XLearnSetValidate(self.handle, c_str(val_path)))

	def fit(self, param, model_path):
		"""Check hyper-parameters, train model, and dump model.

		Parameters
		----------
		param : dict
		  hyper-parameter used by xlearn.
		model_path : str
		  path of model checkpoint.
		"""
		_check_call(_LIB.XLearnFit(self.handle, c_str(model_path)))

	def predict(self, model_path, out_path):
		"""Predict output

        Parameters
        ----------
        model_path : str
          path of model checkpoint.
        out_path : str
          path of output result.
		"""
		_check_call(_LIB.XLearnPredict(self.handle, 
			c_str(model_path), c_str(out_path)))

def create_linear():
	"""
	Create a linear model.
	"""
	model_type = 'linear'
	handle = XLearnHandle()
	_check_call(_LIB.XLearnCreate(c_str(model_type),
		                          ctypes.byref(handle)))
	return XLearn(handle)

def create_fm():
	"""
	Create a factorization machine.
	"""
	model_type = 'fm'
	handle = XLearnHandle()
	_check_call(_LIB.XLearnCreate(c_str(model_type),
		                          ctypes.byref(handle)))
	return XLearn(handle)

def create_ffm():
	"""
	Create a field-aware factorization machine.
	"""
	model_type = 'ffm'
	handle = XLearnHandle()
	_check_call(_LIB.XLearnCreate(c_str(model_type),
		                          ctypes.byref(handle)))
	return XLearn(handle)
