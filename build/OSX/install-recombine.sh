#!/bin/bash -e
# Build recombine from sources and install.
set -u -o xtrace

pushd ../recombine
./doall-macOS.sh
popd
