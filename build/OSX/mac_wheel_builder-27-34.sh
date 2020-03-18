#!/bin/bash
# See mac_wheel_builder.sh for documentation.

p=$1 # Python version

OUTPUTDIR=wheelhouse
TESTDIR=test

rm -fr $TESTDIR
mkdir $TESTDIR

# sudo needed for MacPorts (TODO: parameterise)
sudo $pyexe -m pip install --upgrade pip
sudo $pyexe -m pip install --upgrade wheel
sudo $pyexe -m pip install --upgrade numpy
sudo $pyexe -m pip install --upgrade delocate
sudo $pyexe -m pip install --upgrade virtualenv

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
