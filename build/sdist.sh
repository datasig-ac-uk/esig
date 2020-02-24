#!/bin/bash

# Build source distribution. Run this script from appropriate build subdirectory (Linux or OSX).
# Previously we got this as a .tar.gz from (https://pypi.org/project/esig/#files). Use a virtualenv
# at Python 3.5.5 to avoid depending on the local Python version (although requiring 3.5.5 to be
# available is still not ideal).

rm *.tar.gz
build_dir=$PWD
pushd ../.. # need to run in same directory as setup.py
echo "Python version installed:" 
python --version
# pyenv virtualenv 3.5.5 sdist-env-3.5
# pyenv activate sdist-env-3.5
#   echo "Python version for building source distribution:" 
#   python --version
   pip install cython # not present in GitHub macos-10.15 environment
   python setup.py sdist --dist-dir=$build_dir
#pyenv deactivate
rm -rf esig.egg-info
popd
