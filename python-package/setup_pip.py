# coding: utf-8
"""Setup xlearn package."""
from __future__ import absolute_import

import os
import subprocess
import shutil
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.sdist import sdist

sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

def silent_call(cmd, raise_error=False, error_msg=''):
    try:
        with open(os.devnull, 'w') as shut_up:
            subprocess.check_output(cmd, stderr=shut_up)
            return 0
    except Exception:
        if raise_error:
            raise Exception(error_msg);
        return 1

def copy_files():
    if os.path.isdir('compile'):
        shutil.rmtree('compile')
    src_list = ['demo', 'gtest', 'scripts', 'src']
    for src in src_list:
        dst = 'compile/{}'.format(src)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree('../{}'.format(src), dst)
    shutil.copy('../CMakeLists.txt', 'compile')
    # create empty python-package for cmake
    os.makedirs('compile/python-package')
    open('compile/python-package/CMakeLists.txt', 'w')

def compile_cpp():
    build_path = os.path.join(CURRENT_DIR, 'build_cpp')
    if os.path.isdir(build_path):
        shutil.rmtree(build_path)
    os.makedirs(build_path)
    old_working_dir = os.getcwd()
    os.chdir(build_path)
    src_path = '../compile'
    cmake_cmd = ['cmake', src_path]
    if not os.path.isdir(src_path):
        print('current path: {}'.format(os.getcwd()))
        raise Exception('{} not exists'.format(src_path))
    if os.name == "nt":
        # Windows
        print('Windows users please use github installation.')
    else:
        # Linux, Darwin (OS X), etc.
        silent_call(cmake_cmd, raise_error=True, error_msg='Please install CMake first')
        silent_call(["make", "-j4"], raise_error=True, 
                error_msg='An error has occurred while building xlearn library file')
        suffix_list = ['dylib', 'so']
        for suffix in suffix_list:
            if os.path.isfile('lib/libxlearn_api.{}'.format(suffix)):
                shutil.copy('lib/libxlearn_api.{}'.format(suffix), '../xlearn/')
        if os.path.isdir('../build/lib/xlearn/'):
            for suffix in suffix_list:
                if os.path.isfile('lib/libxlearn_api.{}'.format(suffix)):
                    shutil.copy('lib/libxlearn_api.{}'.format(suffix), '../build/lib/xlearn/')

    os.chdir(old_working_dir)
    
class CustomInstall(install):
    
    def run(self):
        # compile_cpp();
        install.run(self)

class CustomSdist(sdist):
    
    def run(self):
        copy_files()
        sdist.run(self)

class CustomBuildPy(build_py):

    def run(self):
        compile_cpp()
        build_py.run(self)

if __name__ == "__main__":
    setup(name='xlearn',
          version="0.31.a1",
          description="xLearn Python Package",
          maintainer='Chao Ma',
          maintainer_email='mctt90@gmail.com',
          zip_safe=False,
          cmdclass={
              'install': CustomInstall,
              'sdist': CustomSdist,
              'build_py': CustomBuildPy,
          },
          packages=find_packages(),
          # this will use MANIFEST.in during install where we specify additional files,
          # this is the golden line
          include_package_data=True,
          # move data to MANIFEST.in
          license='Apache-2.0',
          classifiers=['License :: OSI Approved :: Apache Software License'],
          url='https://github.com/aksnzhy/xlearn')
