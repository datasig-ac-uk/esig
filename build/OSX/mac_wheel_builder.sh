#!/bin/bash -e
### script to build Python wheels for the esig package, adapted from
### Daniel Wilson-Nunn's original script.
set -o xtrace

p=$1 # Python version

# note: boost needs to be installed, and all versions of python
# listed in python_versions.txt should have been installed via pyenv

TMPDIR=tmp
OUTPUTDIR=output   # location of tested wheels

# setup for pyenv and virtualenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

rm -rf $TMPDIR
rm -rf $OUTPUTDIR
mkdir $TMPDIR
mkdir $OUTPUTDIR

# create and activate a virtualenv for this python version
pyenv virtualenv $p esig_build_env-$p
pyenv activate esig_build_env-$p

# install python dependencies
pip install --upgrade pip
pip install --upgrade wheel
pip install numpy==1.7.0
pip install --upgrade delocate
# build the wheel
pushd .. # circular file path if run from OSX folder
    pip wheel -w OSX/$TMPDIR ..
popd
# combine other dynamic libraries into the wheel to make it portable
delocate-wheel -v $TMPDIR/esig*.whl
# deactivate this virtualenv, then create a fresh one to run tests
pyenv deactivate
pyenv virtualenv $p esig_test_env-$p
pyenv activate esig_test_env-$p
pip install `ls ${TMPDIR}/*.whl`
# run tests
python -m unittest discover -v -s "../../esig/tests"
if [ $? -eq 0 ]
then
    echo "Tests passed."
    mv ${TMPDIR}/esig*.whl $OUTPUTDIR/
else
    echo "Tests failed."
    exit 1
fi
# deactivate this virtualenv
pyenv deactivate

# disable error propagation to avoid sporadic "directory not empty" error
set +e
rm -rf $TMPDIR
