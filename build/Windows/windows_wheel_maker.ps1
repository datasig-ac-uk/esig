Set-PSDebug -Trace 1

# run from C:\data\ directory inside the esig_builder_windows docker container.
# arguments: <python_version_string e.g. python37_64>
# del *.whl
# if not exist "C:\data\output" mkdir C:\data\output
curl -O https://download.microsoft.com/download/E/E/D/EEDF18A8-4AED-4CE0-BEBE-70A83094FC5A/BuildTools_Full.exe
if ( Test-Path -Path 'C:\Program Files (x86)\Microsoft Visual Studio 14.0' -PathType Container ) {
   echo 'Visual Studio 14.0 folder exists.'
} else {
   exit 1
}
.\BuildTools_Full.exe /silent /full /passive

# Boost sources. Use SourceForge instead of boostorg; latter currently returns 403 (Forbidden).
# curl -O https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.zip
curl -L -o boost_1_68_0.zip https://sourceforge.net/projects/boost/files/boost/1.68.0/boost_1_68_0.zip/download
if ($LASTEXITCODE -ne 0) { throw "Boost source download failed." }
Set-PSDebug -Off
Expand-Archive -PassThru .\boost_1_68_0.zip -DestinationPath boost
Set-PSDebug -Trace 1

# Boost binaries.
curl -L -o boost_1_68_0-msvc-14.0-64.exe https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-14.0-64.exe/download

Measure-Command {
   # self-extracting installer - will unpack to C:\local\boost\boost_1_68_0\lib[64,32]-msvc-[version]
   # without /VERYSILENT installer will attempt to open dialog box and silently fail
   Start-Process -Wait -PassThru -FilePath .\boost_1_68_0-msvc-14.0-64.exe -ArgumentList '/VERYSILENT /SP-'
}

# compiler expects boost here
$BOOST_ROOT='.\boost\boost_1_68_0'
mkdir $BOOST_ROOT\x64
mkdir $BOOST_ROOT\x64\lib

ls $BOOST_ROOT
# ABORT ABORT ABORT
exit 0

Move-Item -Path C:\local\boost_1_68_0\lib64-msvc-14.0\*.lib -Destination $BOOST_ROOT\x64\lib

curl -L -O https://www.python.org/ftp/python/3.5.4/python-3.5.4-amd64.exe
$ErrorActionPreference = 'Stop'
$VerbosePreference = 'Continue'
Measure-Command {
   Start-Process -Wait -PassThru -FilePath .\python-3.5.4-amd64.exe -ArgumentList '/quiet'
}

ls C:\Users\runneradmin\AppData\Local\Programs\Python\Python35

$ENV:PATH="C:\Users\runneradmin\AppData\Local\Programs\Python\Python35;C:\Users\runneradmin\AppData\Local\Programs\Python\Python35\Scripts;$ENV:PATH"

# TODO: check 3.5[.4]
python --version

python -m pip install numpy
python -m pip install wheel
python -m pip install delocate
python -m pip install --upgrade setuptools
python -m pip install virtualenv

# build the wheel
pushd ..
   # python.exe -m pip wheel --trusted-host --no-binary -b latest %1
   python.exe -m pip wheel -b Windows/ -w Windows/output/ ..
popd
# don't know how to make the above command fail

# Up to here so far
echo 'All good so far.'
exit 0

# create a virtualenv for testing.
# virtualenv.exe %1
# using the virtualenv python, install the newly created esig wheel
# FOR /F "tokens=* USEBACKQ" %%F IN (`dir /b esig*.whl`) DO %1\Scripts\python.exe -m pip install %%F
# run the tests
# %1\Scripts\python.exe -c "import esig.tests as tests; tests.run_tests(terminate=True)"
# if tests pass, move the wheel to 'output'
# if %ERRORLEVEL% EQU 0 move esig*.whl C:\data\output\
# remove extraneous wheels
# del *.whl
