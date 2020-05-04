#!/bin/bash -e
set -u -o xtrace

### this is the command to be run in the docker container to build the wheels
### from the gzip-ed source.

arch=$1

echo "**************"
echo "Installed Python versions:"
ls /opt/python/
echo "**************"

TMPDIR=tmp         # working folder for pip wheel
OUTPUTDIR=output   # location of tested wheels

rm -rf $OUTPUTDIR
mkdir $OUTPUTDIR

for py in $( cat python_versions.txt ); do
	pyexe=/opt/python/$py/bin/python
	$pyexe -m pip install -U pip virtualenv
	$pyexe -m pip install wheel==0.31.1
	$pyexe -m pip install -U numpy

	rm -rf $TMPDIR
	rm -rf wheelhouse
	mkdir $TMPDIR

	pushd .. # circular file path if run from Linux folder
		$pyexe -m pip wheel -b Linux/$TMPDIR -w Linux/$TMPDIR ..
	popd

	auditwheel repair $TMPDIR/esig*.whl # puts wheel into wheelhouse
	$pyexe -m virtualenv /tmp/$py
	. /tmp/$py/bin/activate
	pip install wheelhouse/esig*.whl
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
	rm -rf /tmp/$py/ ; # TODO: move this into the loop, for each file
done
