#!/bin/bash
# See build-wheels.sh and mac_wheel_builder.sh for documentation.

brew install boost
brew install openssl

# pyexe=/opt/local/bin/python3 # Python 3.4 (MacPorts)
pyexe=/usr/local/bin/python  # Python 2.7
py=$($pyexe --version 2>&1)
p=${py##* }

OUTPUTDIR=wheelhouse
TESTDIR=test

rm -fr $TESTDIR
mkdir $TESTDIR

$pyexe -m pip install --upgrade pip
$pyexe -m pip install --upgrade wheel
$pyexe -m pip install --upgrade numpy
$pyexe -m pip install --upgrade delocate
$pyexe -m pip install --upgrade virtualenv

pushd .. # circular file path if run from OSX folder
   $pyexe -m pip wheel -w OSX/$TESTDIR ..
popd
delocate-wheel -v $TESTDIR/esig*.whl

VENV=esig_test_env-$p
$pyexe -m virtualenv $VENV
   . $VENV/bin/activate
   pip install `ls ${TESTDIR}/*.whl`
   python -c 'import esig.tests as tests; tests.run_tests(terminate=True)'
   if [ $? -eq 0 ]
   then
      echo "Tests passed - copying wheel to $OUTPUTDIR"
      mv ${TESTDIR}/*.whl $OUTPUTDIR/
   else
      echo "Tests failed - will not copy wheel to $OUTPUTDIR"
      exit 1
   fi
deactivate
rm -rf $VENV
rm -fr $TESTDIR
