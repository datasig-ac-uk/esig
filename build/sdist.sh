#!/bin/bash -e
set -u -o xtrace

# Build source distribution. Run from build directory.

pushd .. # need to run in same directory as setup.py
echo "Python version installed (>= 3.5 required):"
python --version
# Cython not present in GitHub macos-10.15 environment

rm -rf esig.egg-info
popd
