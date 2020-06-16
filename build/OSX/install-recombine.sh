#!/bin/bash -e
# Build recombine from sources and install.
set -u -o xtrace

git clone git@github.com:terrylyons/recombine.git
pushd recombine
./doall-macOS.sh
popd
