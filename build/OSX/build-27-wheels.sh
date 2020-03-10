#!/bin/bash
# See build-wheels.sh and mac_wheel_builder.sh for documentation.

brew install boost
brew install openssl

py=$(python --version 2>&1)
p=${py##* }

OUTPUTDIR=wheelhouse
TESTDIR=test
WORKDIR=latest

rm -f $OUTPUTDIR/*
rm -fr $WORKDIR
rm -fr $TESTDIR
mkdir $WORKDIR
mkdir $TESTDIR

python -m pip install --upgrade pip
python -m pip install --upgrade wheel
python -m pip install --upgrade numpy
python -m pip install --upgrade delocate
python -m pip install --upgrade virtualenv

pushd .. # circular file path if run from OSX folder
   pip wheel -b OSX/$WORKDIR -w OSX/$TESTDIR ..
popd
delocate-wheel -v $TESTDIR/esig*.whl

VENV=esig_test_env-$p
python -m virtualenv $VENV
   . $VENV/bin/activate
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
rm -fr $TESTDIR
