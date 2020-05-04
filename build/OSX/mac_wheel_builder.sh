#!/bin/bash -e

### script to build Python wheels for the esig package, adapted from
### Daniel Wilson-Nunn's original script.

p=$1 # Python version

# note: boost needs to be installed, and all versions of python
# listed in python_versions.txt should have been installed via pyenv

OUTPUTDIR=wheelhouse    # where final wheel will be put (don't clean this)
TESTDIR=test            # staging area from which to install the wheel for testing
TMPDIR=tmp              # where wheel will first go, before 'delocate'
WORKDIR=latest          # temp dir used during build

# setup for pyenv and virtualenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Make a couple of clean working directories
rm -fr $WORKDIR
rm -fr $TMPDIR
rm -fr $TESTDIR
mkdir $WORKDIR
mkdir $TMPDIR
mkdir $TESTDIR

# create and activate a virtualenv for this python version
pyenv virtualenv $p esig_build_env-$p
pyenv activate esig_build_env-$p

# install python dependencies
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
    mv ${TESTDIR}/*.whl $OUTPUTDIR/
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
