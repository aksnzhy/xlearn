@echo off
mkdir build && ^
cd build && ^
cmake -G "Visual Studio 15 Win64" ../ && ^ 
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 && ^
MSBuild xLearn.sln /p:Configuration=Release && ^
REM the under two lines run correct only if the project on C(drive disk), you can comment this two lines.
cd python-package && ^
python setup.py bdist_wheel -p win_amd64 --python-tag cp36 --py-limited-api cp36m