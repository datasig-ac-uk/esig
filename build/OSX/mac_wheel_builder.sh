#!/bin/bash

### script to build Python wheels for the esig package, adapted from
### Daniel Wilson-Nunn's original script.
#
# - create and activate a virtual environment
# - install python dependencies
# - compile the Python wheel
# - "delocate" the wheel, i.e. combine other dependencies, to give a portable binary
# - create another virtual environment and install the newly created wheel
# - run the esig unit tests
# - if the tests pass, place wheel in wheelhouse subdirectory

p=$1 # Python version

# note boost needs to be installed, and all versions of python
# listed in python_versions.txt should have been installed via pyenv

# OUTPUTDIR is where the final wheel will be put
OUTPUTDIR=wheelhouse
# TESTDIR is a staging area from which to install the wheel for testing
TESTDIR=test
# TMPDIR is where the wheel will go when first build, before 'delocate'
TMPDIR=tmp
# WORKDIR is a temporary directory used in the build process
WORKDIR=latest

# setup for pyenv and virtualenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Make a couple of clean working directories
rm -f $WORKDIR/*
rm -fr $WORKDIR
rm -fr $TMPDIR
rm -fr $TESTDIR
mkdir $WORKDIR
mkdir $TMPDIR
mkdir $TESTDIR

# create and activate a virtualenv for this python version
if ! pyenv virtualenv $p esig_build_env-$p
then
    echo "Launching virtualenv for build failed."
    exit 1
fi
pyenv activate esig_build_env-$p

# install python dependencies
# assume pip is provided by the host environment at a suitable version
pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade numpy
pip install --upgrade delocate
# build the wheel
pushd .. # circular file path if run from OSX folder
    pip wheel -b OSX/$WORKDIR -w OSX/$TMPDIR ..
popd
# combine other dynamic libraries into the wheel to make it portable
delocate-wheel -w $TESTDIR -v $TMPDIR/esig*.whl
# deactivate this virtualenv, then create a fresh one to run tests
pyenv deactivate
pyenv virtualenv $p esig_test_env-$p
pyenv activate esig_test_env-$p
pip install `ls ${TESTDIR}/*.whl`
# run tests
python -c 'import esig.tests as tests; tests.run_tests(terminate=True)'
if [ $? -eq 0 ]
then
    echo "Tests passed - copying wheel to $OUTPUTDIR"
    mv ${TESTDIR}/*.whl $OUTPUTDIR
else
    echo "Tests failed - will not copy wheel to $OUTPUTDIR"
    exit 1
fi
# deactivate this virtualenv
pyenv deactivate

# cleanup
rm -fr $WORKDIR
rm -fr $TMPDIR
rm -fr $TESTDIR
