@echo off
mkdir build && cd build && ^
cmake -G "Visual Studio 15 Win64" ../ && "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 && ^
MSBuild xLearn.sln /p:Configuration=Release && ^
cd python-package && ^
python setup.py install