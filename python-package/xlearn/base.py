# coding: utf-8
import sys
import os
import ctypes
from .libpath import find_lib_path

class XLearnError(Exception):
	"""Error thrown by xlearn trainer"""
	pass

def _load_lib():
	"""Load xlearn shared library"""
	lib_path = find_lib_path()
	if len(lib_path) == 0:
		return None
	lib = ctypes.cdll.LoadLibrary(lib_path[0])
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

# type definitions
XLearnHandle = ctypes.c_void_p

if sys.version_info[0] < 3:
	def c_str(string):
		"""Create ctypes char * from a Python string.

		Parameters
		----------
		  string : string type
		     Pyrhon string.

		Returns
		-------
		str : c_char_p
		    A char pointer that can be passed to C API.

		Examples
		--------
		>>> x = c_str("Hello, world!")
		>>> print x.value
		Hello, world!
		"""
		return ctypes.c_char_p(string)
else:
	def c_str(string):
		"""Create ctypes char * from a Python string.

		Parameters
		----------
		  string : string type
		     Pyrhon string.

		Returns
		-------
		str : c_char_p
		    A char pointer that can be passed to C API.

		Examples
		--------
		>>> x = c_str("Hello, world!")
		>>> print x.value
		Hello, world!
		"""
		return ctypes.c_char_p(string.encode('utf-8'))
