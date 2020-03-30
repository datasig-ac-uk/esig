Set-PSDebug -Trace 1

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

# Boost sources. Unpacks to "boost_1_68_0" subfolder of DestinationPath.
curl -L -o boost_1_68_0.zip https://sourceforge.net/projects/boost/files/boost/1.68.0/boost_1_68_0.zip/download
if ($LASTEXITCODE -ne 0) { throw "Boost source download failed." }
Set-PSDebug -Off
Expand-Archive .\boost_1_68_0.zip -DestinationPath C:\boost
Set-PSDebug -Trace 1

# Tell install_helpers.py where the Boost sources were installed.
$ENV:BOOST_ROOT='C:\boost\boost_1_68_0\'

# Boost binaries.
# curl -L -o boost_1_68_0-msvc-14.0-64.exe https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-14.0-64.exe/download
curl -L -o boost_1_68_0-msvc-14.1-64.exe https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-14.1-64.exe/download

# self-extracting installer - will unpack to C:\local\boost\boost_1_68_0\lib[64,32]-msvc-[version]
# without /VERYSILENT installer will attempt to open dialog box and silently fail
Start-Process -Wait -PassThru -FilePath .\boost_1_68_0-msvc-14.1-64.exe -ArgumentList '/VERYSILENT /SP-'

mkdir $ENV:BOOST_ROOT\x64
mkdir $ENV:BOOST_ROOT\x64\lib

Move-Item -Path C:\local\boost_1_68_0\lib64-msvc-14.1\*.lib -Destination $ENV:BOOST_ROOT\x64\lib

curl -L -O https://www.python.org/ftp/python/3.5.4/python-3.5.4-amd64.exe
$ErrorActionPreference = 'Stop'
$VerbosePreference = 'Continue'
Start-Process -Wait -PassThru -FilePath .\python-3.5.4-amd64.exe -ArgumentList '/quiet'

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
   python.exe -m pip wheel -b Windows/ -w Windows/output/ ..
popd
if ($LASTEXITCODE -ne 0) { throw "pip wheel failed." }

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
