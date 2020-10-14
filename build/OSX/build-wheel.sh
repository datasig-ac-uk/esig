#!/bin/bash -e
# Top-level script for building esig for MacOS.
# $1: Python version
set -u -o xtrace

rm -rf ~/.pyenv/versions # for reproducibility

pushd ../recombine
./doall-macOS.sh
popd

function brew_maybe_install {
   brew list $1 &>/dev/null || brew install $1
}

brew_maybe_install boost
brew_maybe_install pyenv
brew_maybe_install pyenv-virtualenv
# Explicitly request openssl 1.1
brew_maybe_install openssl@1.1 # required for Python 3.7

# Python versions.
eval "$(pyenv init -)"

# see https://github.com/pyenv/pyenv/wiki/common-build-problems
CFLAGS="-I$(brew --prefix openssl)/include" \
LDFLAGS="-L$(brew --prefix openssl)/lib" \
pyenv install --skip-existing $1

./mac_wheel_builder.sh $1
