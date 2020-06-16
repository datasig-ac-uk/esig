#!/bin/bash -e
# Top-level script for building esig for MacOS.
# $1: Python version
set -u -o xtrace

rm -rf ~/.pyenv/versions # for reproducibility

pushd ../recombine
./doall-macOS.sh
popd

brew install boost
brew install pyenv
brew install pyenv-virtualenv
brew install openssl # required for Python 3.7

# Python versions.
eval "$(pyenv init -)"

# see https://github.com/pyenv/pyenv/wiki/common-build-problems
CFLAGS="-I$(brew --prefix openssl)/include" \
LDFLAGS="-L$(brew --prefix openssl)/lib" \
pyenv install --skip-existing $1

./mac_wheel_builder.sh $1
