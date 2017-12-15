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
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # make pythonpack hack: copy this directory one level upper for setup.py
    dll_path = [curr_path, os.path.join(curr_path, '../../lib/'),
                os.path.join(curr_path, './lib/'),
                os.path.join(sys.prefix, 'xlearn')]
    if sys.platform == 'win32':
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(curr_path, '../../windows/x64/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/x64/Release/'))
        else:
            dll_path.append(os.path.join(curr_path, '../../windows/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/Release/'))
        dll_path = [os.path.join(p, 'xlearn_api.dll') for p in dll_path]
    elif sys.platform.startswith('linux'):
        dll_path = [os.path.join(p, 'libxlearn_api.so') for p in dll_path]
    elif sys.platform == 'darwin':
        dll_path = [os.path.join(p, 'libxlearn_api.dylib') for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

    # From github issues, most of installation errors come from machines w/o compilers
    if not lib_path:
        raise XLearnLibraryNotFound(
            'Cannot find xlearn Library in the candidate path'
            )
    return lib_path
