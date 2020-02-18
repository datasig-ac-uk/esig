#!/bin/bash
# Top-level script for building esig for MacOS.
# Python wheels will be created for Python versions specified in python_versions.txt.

# esig needs `boost`. Packages `pyenv` and `pyenv-virtualenv` are needed for the script below
# to build for multiple Python versions. Python 3.7 requires `openssl`.
brew install boost
brew install pyenv
brew install pyenv-virtualenv
brew install openssl

# Get esig sources.
rm *.tar.gz
pushd ../.. # need to run in same directory as setup.py
python setup.py sdist --dist-dir=build/OSX
rm -rf esig.egg-info
popd

# Wheels that pass their tests will be placed here.
rm -rf wheelhouse

# How persistent is this URL?
wget --no-clobber https://files.pythonhosted.org/packages/7c/4f/d0dd6fd1054110efc73ae745036fb57e37017ed45ed449bf472f92197024/esig-0.6.31.tar.gz
. install_all_python_versions.sh

# Build the esig wheels.
for p in $(cat python_versions.txt); do
   . mac_wheel_builder.sh $p
done
