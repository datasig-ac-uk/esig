#!/bin/bash -e
set -u -o xtrace
# See mac_wheel_builder.sh for documentation.

path_ext=$1 # Python 3.4 needs explicit path for delocate
run_as=$2 # either "" or "sudo"

TMPDIR=tmp

rm -rf $TMPDIR
mkdir $TMPDIR

# sudo needed for MacPorts
$run_as $pyexe -m pip install --upgrade pip
$run_as $pyexe -m pip install --upgrade wheel
$run_as $pyexe -m pip install --upgrade numpy
$run_as $pyexe -m pip install --upgrade delocate
$run_as $pyexe -m pip install --upgrade virtualenv

pushd .. # circular file path if run from OSX folder
   $pyexe -m pip wheel -w OSX/$TMPDIR ..
popd
PATH=$PATH:$path_ext # to find delocate-wheel
delocate-wheel -v $TMPDIR/esig*.whl

VENV=esig_test_env
rm -rf $VENV
$pyexe -m virtualenv $VENV
   . $VENV/bin/activate
   pip install `ls ${TMPDIR}/*.whl`
   python -c 'import esig.tests as tests; tests.run_tests(terminate=True)'
   if [ $? -eq 0 ]
   then
      echo "Tests passed."
      mv ${TMPDIR}/esig*.whl output
   else
      echo "Tests failed."
      exit 1
   fi
deactivate
rm -rf $VENV
rm -rf $TMPDIR
