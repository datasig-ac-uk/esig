#!/bin/bash -e
set -u -o xtrace

### this is the command to be run in the docker container to build the wheels
### from the gzip-ed source.

py=$1
arch=$2

pushd ../recombine
./doall-linux.sh lib64
popd

# recombine build step must leave library here for linking
libdir=${HOME}/lyonstech/lib64
ls -la ${libdir} # check exists

TMPDIR=tmp         # working folder for pip wheel
OUTPUTDIR=output   # location of tested wheels

rm -rf $TMPDIR
rm -rf wheelhouse
rm -rf $OUTPUTDIR
mkdir $OUTPUTDIR
mkdir $TMPDIR

pyexe=/opt/python/$py/bin/python
ls -la /opt/python # see what Python versions there are
$pyexe -m pip install -U pip virtualenv
$pyexe -m pip install wheel==0.34.2 # need at least this version for auditwheel 3.1.1 (Python 3.7)
$pyexe -m pip install -U numpy

export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH}"
pushd .. # circular file path if run from Linux folder
	$pyexe -m pip wheel -b Linux/$TMPDIR -w Linux/$TMPDIR ..
popd

auditwheel show $TMPDIR/esig*.whl	# useful to see dependencies
auditwheel repair $TMPDIR/esig*.whl # puts wheel into wheelhouse

$pyexe -m virtualenv /tmp/$py
. /tmp/$py/bin/activate
pip install wheelhouse/esig*.whl
python -c 'print("hello")'
python -c 'import esig.tests as tests'
python -c 'import esig.tests as tests; tests.run_tests(terminate=True)'
if [ $? -eq 0 ]
then
	echo "Tests passed - copying wheel to $OUTPUTDIR"
	mv wheelhouse/esig*.whl $OUTPUTDIR/
else
	echo "Tests failed"
	exit 1
fi
deactivate
rm -rf /tmp/$py/ # TODO: move this into the loop, for each file
