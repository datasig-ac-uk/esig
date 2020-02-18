# Build source distribution (Powershell version). Run this script from appropriate build subdirectory 
# (i.e. Windows). Needs to be kept in sync with bash version.

rm *.tar.gz
$build_dir=$PWD
pushd ../.. # need to run in same directory as setup.py
python setup.py sdist --dist-dir=$build_dir
rm esig.egg-info -r -fo
popd
