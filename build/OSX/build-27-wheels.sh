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

pushd .. # circular file path if run from OSX folder
   pip wheel -b OSX/$WORKDIR -w OSX/$TMPDIR ..
popd
delocate-wheel -w $TESTDIR -v $TMPDIR/esig*.whl

VENV=esig_test_env-$p
python -m virtualenv $VENV
   $VENV/bin/activate
   pip install `ls ${TESTDIR}/*.whl`
   python -c 'import esig.tests as tests; tests.run_tests(terminate=True)'
   if [ $? -eq 0 ]
   then
      echo "Tests passed - copying wheel to $OUTPUTDIR"
      mv ${TESTDIR}/*.whl $OUTPUTDIR
   else
      echo "Tests failed - will not copy wheel to $OUTPUTDIR"
      exit 1
   fi
deactivate
rm -rf $VENV

rm -fr $WORKDIR
rm -fr $TMPDIR
rm -fr $TESTDIR
