#!/bin/bash

# Build source distribution. Run this script from appropriate build subdirectory (Linux or OSX).
# Previously we got this as a .tar.gz from (https://pypi.org/project/esig/#files). We depend on a
# local Python installation to build sdist; the CI environment takes care of this.

rm -f *.tar.gz
build_dir=$PWD
pushd ../.. # need to run in same directory as setup.py
echo "Python version installed (>= 3.5 required):"
python --version
   pip install cython # not present in GitHub macos-10.15 environment
   python setup.py sdist --dist-dir=$build_dir
rm -rf esig.egg-info
popd
