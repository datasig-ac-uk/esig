#!/bin/bash -e
# Build recombine from sources and install.
set -u -o xtrace

rm -rf recombine
# git config user.email "email"
git config --global url."git@github.com:".insteadOf https://github.com/
git config user.name "rolyp"
git clone git@github.com:terrylyons/recombine.git

pushd recombine
./doall-macOS.sh
popd
