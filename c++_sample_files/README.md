#  C++ Sample Files

It contains following files. You are allowed to create the files of your own. But player file must be named at student_agent.py

- student_agent_cpp.py - Serves as a wrapper between python and c++.
- student_agent.cpp - Can be used to write your c++ code.
- CMakeLists.txt - CMake file

## Dependencies
pybind11

## Installation
```sh
pip install pybind11
```

## Setting up the C++ Agent

```sh
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
 -DCMAKE_C_COMPILER=gcc \
 -DCMAKE_CXX_COMPILER=g++
```
Run:

```sh
make
cd ..
```

## Running for a C++ program.

You need to use student_cpp rather than student for this case.

```sh
python gameEngine.py --mode aivai --circle random --square student_cpp
```

