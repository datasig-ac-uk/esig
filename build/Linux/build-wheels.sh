#!/bin/bash

# Top-level script for building esig for Linux on MacOS or Linux.
# Python wheels will be created for 64-bit (x86_64) and for 32-bit (i686), for Python versions
# specified in python_versions.txt.

# Build Docker images. 
# This will be quick if there is an image already cached.
for arch in i686 x86_64
do
   docker build -t esig_builder_linux_${arch} -f Dockerfile_${arch}.dockerfile .
done

# Get esig sources.
source ../sdist.sh

# Wheels that pass their tests will be placed here.
rm -rf wheelhouse

# Build the esig wheels.
# The `linux_wheel_maker.sh` script is run inside the docker container, and performs the steps to 
# build the esig binary wheel, run tests, and if the tests are successful, copy the wheel to the 
# `wheelhouse` directory.
for arch in i686 x86_64
do 
   docker run --rm -v ${PWD}:/data esig_builder_linux_${arch} "source ~/.bashrc; cd /data; source linux_wheel_maker.sh $arch"
done
