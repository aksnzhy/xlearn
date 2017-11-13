# coding: utf-8
import sys
import os
import ctypes

def _load_lib():
	"""Load xlearn library"""
	lib_path = find_lib_path()
	if len(lib_path) == 0:
		return None
	lib = ctypes.cdll.Loadlibrary(lib_path[0])
	return lib

# load the xlearn library globally
_LIB = _load_lib()