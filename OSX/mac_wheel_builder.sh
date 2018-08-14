#!/bin/bash

### script to build python wheels for the esig package, adapted from
### Daniel Wilson-Nunn's original script.

# note boost needs to be installed, and all versions of python
# listed in python_versions.txt should have been installed via pyenv


WORKDIR=latest
OUTPUTDIR=output
TMPDIR=tmp

# make the output directory if it isn't already there
mkdir -p $OUTPUTDIR

# setup for pyenv and virtualenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# loop over all python versions
for p in $(cat python_versions.txt); do

# Make a couple of clean working directories
    rm -fr $WORKDIR
    rm -fr $TMPDIR
    mkdir $WORKDIR
    mkdir $TMPDIR

# create and activate a virtualenv for this python version
    pyenv virtualenv $p esig_build_env-$p
    pyenv activate esig_build_env-$p

# install python dependencies
    pip install --upgrade pip
    pip install --upgrade wheel
    pip install --upgrade numpy
    pip install --upgrade delocate
# build the wheel
    pip wheel --no-binary -b $WORKDIR -w $TMPDIR esig*.tar.gz
# combine other dynamic libraries into the wheel to make it portable
    delocate-wheel -w $OUTPUTDIR -v $TMPDIR/esig*.whl
    # deactivate this virtualenv
    source deactivate
done

# cleanup
rm -fr $WORKDIR
rm -fr $TMPDIR
