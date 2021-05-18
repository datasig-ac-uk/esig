#!/bin/bash -e
set -u -o xtrace

### this is the command to be run in the docker container to build the wheels
### from the gzip-ed source.

py=$1
arch=$2 # {i686, x86_64}

if [[ $arch == "i686" ]]
then
   libdir="lib"
elif [[ "$arch" == "x86_64" ]]
then
   libdir="lib64"
else
   exit 1
fi

pushd ../recombine
./doall-linux.sh $libdir $arch
popd

TMPDIR=tmp         # working folder for pip wheel
OUTPUTDIR=output   # location of tested wheels

rm -rf $TMPDIR
rm -rf wheelhouse
rm -rf $OUTPUTDIR
mkdir -p $OUTPUTDIR
mkdir $TMPDIR

pyexe=/opt/python/$py/bin/python
ls -la /opt/python # see what Python versions there are
$pyexe -m pip install -U pip virtualenv
$pyexe -m pip install -U 'wheel>=0.34.2' # need at least this version for auditwheel 3.1.1 (Python 3.7)
$pyexe -m pip install  --only-binary :all: 'numpy==1.7.0'

# recombine build step will have left library here
recombine_install_dir=${HOME}/lyonstech
ls -la ${recombine_install_dir}/$libdir # assert exists
pushd ${recombine_install_dir}/bin # set_env_test_recombine.sh expects to run at its location
	source set_env_test_recombine.sh $libdir # set LD_LIBRARY_PATH and OMP options
popd

echo $ESIG_WITH_RECOMBINE
pushd .. # circular file path if run from Linux folder
	$pyexe -m pip wheel -w Linux/$TMPDIR ..
popd

auditwheel show $TMPDIR/esig*.whl	# useful to see dependencies
auditwheel repair $TMPDIR/esig*.whl # will leave wheel in wheelhouse

$pyexe -m virtualenv /tmp/$py
source /tmp/$py/bin/activate
pip install wheelhouse/esig*.whl
python -m unittest discover -v -s /data/esig/tests
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
