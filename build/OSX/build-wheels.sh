#!/bin/bash
# Top-level script for building esig for MacOS.
# Python wheels will be created for Python versions specified in python_versions.txt.

# esig needs `boost`. Packages `pyenv` and `pyenv-virtualenv` are needed for the script below
# to build for multiple Python versions. Python 3.7 requires `openssl`.
brew install boost
brew install pyenv
brew install pyenv-virtualenv
brew install openssl

# Python versions.
source install_all_python_versions.sh

# Get esig sources.
source ../sdist.sh build/OSX

# Wheels that pass their tests will be placed here.
rm -rf wheelhouse


# Build the esig wheels.
for p in $(cat python_versions.txt); do
   . mac_wheel_builder.sh $p
done
