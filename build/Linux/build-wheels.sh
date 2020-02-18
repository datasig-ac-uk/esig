#!/bin/bash

# Top-level script for building esig for Linux on MacOS or Linux.
# Python wheels will be created for 64-bit (x86_64) and for 32-bit (i686), for python versions
# specified in python_versions.txt.

# Wheels that pass their tests will be placed here.
rm -rf wheelhouse

# Build Docker images. 
# This will be quick if there is an image already cached.
docker build -t esig_builder_linux_i686 -f Dockerfile_i686.dockerfile .
docker build -t esig_builder_linux_x86_64 -f Dockerfile_x86_64.dockerfile .

# Get esig sources.
# Previously we got this as a .tar.gz from (https://pypi.org/project/esig/#files), but now we build
# from the local sources. We are now dependent on the local Python version to build the source
# distribution; maybe we should build the wheels directly from sources instead?
rm *.tar.gz
pushd ../.. # need to run in same directory as setup.py
python setup.py sdist --dist-dir=build/Linux
rm -rf esig.egg-info
popd

# Build the esig wheels.
# The `linux_wheel_maker.sh` script is run inside the docker container, and performs the steps to 
# build the esig binary wheel, run tests, and if the tests are successful, copy the wheel to the 
# `wheelhouse` directory.
for arch in i686 # x86_64 currently broken
do 
   docker run --rm -v ${PWD}:/data esig_builder_linux_${arch} "source ~/.bashrc; cd /data; source linux_wheel_maker.sh $arch"
done
