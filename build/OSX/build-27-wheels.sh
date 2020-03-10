#!/bin/bash

brew install boost
brew install openssl

# Wheels that pass their tests will be placed here.
rm -rf wheelhouse

p=2.7.17

OUTPUTDIR=wheelhouse
TESTDIR=test
TMPDIR=tmp
WORKDIR=latest

mkdir -p $OUTPUTDIR
rm -fr $WORKDIR
rm -fr $TMPDIR
rm -fr $TESTDIR
mkdir $WORKDIR
mkdir $TMPDIR
mkdir $TESTDIR

python -m pip install --upgrade pip
python -m pip install --upgrade wheel
python -m pip install --upgrade numpy
python -m pip install --upgrade delocate

pip wheel -b $WORKDIR -w $TMPDIR ../..
delocate-wheel -w $TESTDIR -v $TMPDIR/esig*.whl
# deactivate this virtualenv, then create a fresh one to run tests
# pyenv deactivate
# pyenv virtualenv $p esig_test_env-$p
# pyenv activate esig_test_env-$p
# pip install `ls ${TESTDIR}/*.whl`
# run tests
# python -c 'import esig.tests as tests; tests.run_tests(terminate=True)'
# if [ $? -eq 0 ]
# then
#    echo "Tests passed - copying wheel to $OUTPUTDIR"
#    mv ${TESTDIR}/*.whl $OUTPUTDIR
# else
#    echo "Tests failed - will not copy wheel to $OUTPUTDIR"
#     exit 1
# fi
# deactivate this virtualenv
# pyenv deactivate

rm -fr $WORKDIR
rm -fr $TMPDIR
rm -fr $TESTDIR
