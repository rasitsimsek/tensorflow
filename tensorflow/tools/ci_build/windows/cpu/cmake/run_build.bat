:: This script assumes the standard setup on tensorflow Jenkins windows machines.
:: It is NOT guaranteed to work on any other machine. Use at your own risk!
::
:: REQUIREMENTS:
:: * All installed in standard locations:
::   - JDK8, and JAVA_HOME set.
::   - Microsoft Visual Studio 2015 Community Edition
::   - Msys2
::   - Anaconda3
::   - CMake
:: * Before running this script, you have to set BUILD_CC_TESTS and BUILD_PYTHON_TESTS
::   variables to either "ON" or "OFF".
:: * Either have the REPO_ROOT variable set, or run this from the repository root directory.
:: Import all bunch of variables Visual Studio needs.
::CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build\vcvars64.bat"

:: Check and set REPO_ROOT
SET REPO_ROOT=%cd%..\..\..\..\..\..\..
IF [%REPO_ROOT%] == [] (
  SET REPO_ROOT=%cd%
)



:: Turn echo back on, above script turns it off.
ECHO ON

:: Set environment variables to be shared between runs. Do not override if they
:: are set already.

IF DEFINED CMAKE_EXE (ECHO CMAKE_EXE is set to %CMAKE_EXE%) ELSE (SET CMAKE_EXE="C:\Program Files (x86)\CMake\bin\cmake.exe")
IF DEFINED SWIG_EXE (ECHO SWIG_EXE is set to %SWIG_EXE%) ELSE (SET SWIG_EXE="C:\Development\dev\test_projects\deeplearning\swigwin-3.0.12\swig.exe")
IF DEFINED PY_EXE (ECHO PY_EXE is set to %PY_EXE%) ELSE (SET PY_EXE="C:\Users\rsimsek\AppData\Local\Programs\Python\Python35\python.exe")
IF DEFINED PY_LIB (ECHO PY_LIB is set to %PY_LIB%) ELSE (SET PY_LIB="C:\Development\dev\test_projects\Python-3.5.4\PCbuild\amd64\python35_d.lib")

SET CMAKE_DIR=%REPO_ROOT%\tensorflow\contrib\cmake
SET MSBUILD_EXE="msbuild.exe"

:: Run cmake to create Visual Studio Project files.
%CMAKE_EXE% %CMAKE_DIR% -A x64 -DSWIG_EXECUTABLE=%SWIG_EXE% -DPYTHON_EXECUTABLE=%PY_EXE% -DCMAKE_BUILD_TYPE=Debug -DPYTHON_LIBRARIES=%PY_LIB% -DPYTHON_INCLUDE_DIR="C:\Users\rsimsek\AppData\Local\Programs\Python\Python35\include" -DNUMPY_INCLUDE_DIR="C:\Users\rsimsek\AppData\Roaming\Python\Python35\site-packages\numpy\core\include" -DSWIG_DIR="C:\Development\dev\test_projects\deeplearning\swigwin-3.0.12\Lib" -Dtensorflow_BUILD_PYTHON_TESTS=%BUILD_PYTHON_TESTS% -Dtensorflow_BUILD_CC_TESTS=%BUILD_CC_TESTS%

:: Run msbuild in the resulting VS project files to build a pip package.
::%MSBUILD_EXE% /p:Configuration=Debug /maxcpucount:32 tf_python_build_pip_package.vcxproj
%MSBUILD_EXE% /p:Configuration=Debug /maxcpucount:32 tensorflow.vcxproj
