"""Setup xlearn package."""
import sys
import os
import shutil
from setuptools import setup, Distribution, find_packages

class BinaryDistribution(Distribution):
  """Overrides Distribution class to bundle platform-specific binaries"""
  # pylint: disable=R0201
  def has_ext_modules(self):
    """Has an extension module"""
    return True

LIBPATH_PY = os.path.abspath('./xlearn/libpath.py')
LIBPATH = {'__file__': LIBPATH_PY}
exec(compile(open(LIBPATH_PY, "rb").read(), LIBPATH_PY, 'exec'), LIBPATH, LIBPATH)

# Path for C/C++ libraries
LIB_PATH = LIBPATH['find_lib_path']()

setup(name='xlearn',
	  version=open(os.path.join(CURRENT_DIR, 'xlearn/VERSION')).read().strip(),
	  description="xLearn Python Package",
      maintainer='Chao Ma',
      maintainer_email='mctt90@gmail.com',
      license='Apache-2.0',
      classifiers=['License :: OSI Approved :: Apache Software License'],
      url='https://github.com/aksnzhy/xlearn')