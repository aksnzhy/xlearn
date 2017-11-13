# coding: utf-8
"""Find the path to xlearn dynamic library files."""

import os
import platform
import sys

class XLearnLibraryNotFound(Exception):
	"""Error thrown by when xlearn is not found"""
	pass

def find_lib_path():
	"""Find the path to xlearn dynamic library files.

	Returns
	-------
	lib_path: list(string)
	   List of all found library path to xlearn
	"""
	