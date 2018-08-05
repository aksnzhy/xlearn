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
