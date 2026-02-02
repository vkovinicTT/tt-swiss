---
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, Delete
---

I want you to do a full clean build of the tt-xla repository. Please remove all of the following files/folders first

build/*
.cache/
python_package/pjrt_plugin_tt/__pycache__/
python_package/pjrt_plugin_tt/native.cpython-311-x86_64-linux-gnu.so
python_package/pjrt_plugin_tt/pjrt_plugin_tt.so
python_package/torch_plugin_tt/__pycache__/
python_package/tt_torch/__pycache__/
python_package/tt_torch/backend/__pycache__/
python_package/ttxla_tools/__pycache__/
third_party/tt-mlir/tmp
third_party/tt-mlir/install
third_party/tt-mlir/src/tt-mlir/build/*
third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build/*
third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/tmp
venv/bin/
venv/include/
venv/lib/
venv/lib64
venv/pyvenv.cfg

Then perform source venv/activate (to init and activate enviroment). Check if $1 is one of the debug or release. If none, do a debug build.

Create makefile cmake --preset [debug|release]
cmake --build build -- -j20