#!/bin/bash

# Build source distribution. Run this script from appropriate build subdirectory (Linux or OSX).
# Previously we got this as a .tar.gz from (https://pypi.org/project/esig/#files). This leaves us
# dependent on the local Python version to build the source distribution, so at some point we might 
# want to worry about that.

rm *.tar.gz
build_dir=$PWD
echo WIBBLE: $build_dir
pushd ../.. # need to run in same directory as setup.py
python setup.py sdist --dist-dir=$build_dir
rm -rf esig.egg-info
popd
