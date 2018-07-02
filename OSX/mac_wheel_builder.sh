#!/bin/bash

### script to build python wheels for the esig package, by Daniel Wilson-Nunn

# note boost preinstalled!

# do this in virtualenvs

WORKDIR=${HOME}/esig_mac_builds/latest

# Make folder to save new wheels
mkdir -p $WORKDIR && cd "$_"
mkdir -p "fixed"

# Python 2.7
pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade numpy
pip install --upgrade delocate
pip wheel --no-binary -b $WORKDIR esig
delocate -w fixed -v

# Python 3.4
python3.4 -m pip install --upgrade pip
python3.4 -m pip install --upgrade wheel
python3.4 -m pip install --upgrade numpy
python3.4 -m pip install --upgrade delocate
python3.4 -m pip wheel --no-binary -b $WORKDIR esig

# Python 3.5
python3.5 -m pip install --upgrade pip
python3.5 -m pip install --upgrade wheel
python3.5 -m pip install --upgrade numpy
python3.5 -m pip install --upgrade delocate
python3.5 -m pip wheel --no-binary -b $WORKDIR esig

# Python 3.6
pip3 install --upgrade pip
pip3 install --upgrade wheel
pip3 install --upgrade numpy
pip3 install --upgrade delocate
pip3 wheel --no-binary -b $WORKDIR esig
