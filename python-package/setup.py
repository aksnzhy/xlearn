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
"""Setup xlearn package."""
from __future__ import absolute_import
import sys
import os
from setuptools import setup, find_packages
sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

libpath_py = os.path.join(CURRENT_DIR, 'xlearn/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

LIB_PATH = [os.path.relpath(libfile, CURRENT_DIR) for libfile in libpath['find_lib_path']()]
print("Install libxlearn_api from: %s" % LIB_PATH)

setup(name='xlearn',
      version=open(os.path.join(CURRENT_DIR, 'xlearn/VERSION')).read().strip(),
      description="xLearn Python Package",
      maintainer='Chao Ma',
      maintainer_email='mctt90@gmail.com',
      zip_safe=False,
      packages=find_packages(),
      # this will use MANIFEST.in during install where we specify additional files,
      # this is the golden line
      include_package_data=True,
      install_requires=[
            "numpy", 
            "scipy"
      ],
      data_files=[('xlearn', LIB_PATH)],
      license='Apache-2.0',
      classifiers=['License :: OSI Approved :: Apache Software License'],
      url='https://github.com/aksnzhy/xlearn')