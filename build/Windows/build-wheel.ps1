param([string] $vs_version,            # {14.1}
      [string] $arch,                  # {32, 64}
      [string] $py_install_dir,
      [string] $py_installer)

Set-PSDebug -Trace 1

if ($vs_version -eq "14.1") {
   # Use pre-installed Visual Studio
} else {
   exit 1
}

if ($arch -eq "32") {
   $boost_platform_dir="win32"
   $conda_subdir="win-32"
} elseif ($arch -eq "64") {
   $boost_platform_dir="x64"
   $conda_subdir="win-64"
} else {
   exit 1
}

pushd ..\recombine
.\doall-windows.ps1 $conda_subdir
popd

$boost_lib_dir="lib$arch-msvc-$vs_version"
$boost_installer="boost_1_68_0-msvc-$vs_version-$arch.exe"

# Boost sources. Unpacks to "boost_1_68_0" subfolder of DestinationPath.
curl -L -o boost_1_68_0.zip https://sourceforge.net/projects/boost/files/boost/1.68.0/boost_1_68_0.zip/download
if ($LASTEXITCODE -ne 0) { throw "Boost source download failed." }
Set-PSDebug -Off
Expand-Archive .\boost_1_68_0.zip -DestinationPath C:\boost
Set-PSDebug -Trace 1

# Tell install_helpers.py where the Boost sources were installed.
$ENV:BOOST_ROOT='C:\boost\boost_1_68_0\'

# Boost binaries.
curl -L -o $boost_installer https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/$boost_installer/download

# self-extracting installers - will unpack to C:\local\boost\boost_1_68_0\$boost_lib_dir
# without /VERYSILENT installer will attempt to open dialog box and silently fail
Start-Process -Wait -PassThru -FilePath .\$boost_installer -ArgumentList '/VERYSILENT /SP-'

mkdir $ENV:BOOST_ROOT\$boost_platform_dir
mkdir $ENV:BOOST_ROOT\$boost_platform_dir\lib

Move-Item -Path C:\local\boost_1_68_0\$boost_lib_dir\*.lib -Destination $ENV:BOOST_ROOT\$boost_platform_dir\lib

$py_installer_name="install-python" + [System.IO.Path]::GetExtension($py_installer)
curl -L -o $py_installer_name https://www.python.org/ftp/python/$py_installer
$ErrorActionPreference = 'Stop'
$VerbosePreference = 'Continue'
Start-Process -Wait -PassThru -FilePath .\$py_installer_name -ArgumentList '/quiet'

$py_exe=$py_install_dir + "\python.exe"
echo $py_exe
echo (Invoke-Expression "$py_exe --version")

Invoke-Expression "$py_exe -m pip install numpy"
Invoke-Expression "$py_exe -m pip install wheel"
Invoke-Expression "$py_exe -m pip install delocate"
Invoke-Expression "$py_exe -m pip install --upgrade setuptools"
Invoke-Expression "$py_exe -m pip install virtualenv"

# build the wheel
pushd ..
   cp ~\lyonstech\bin\recombine.dll ..\esig\recombine.dll
   cp ~\lyonstech\bin\libiomp5md.dll ..\esig\libiomp5md.dll
   Invoke-Expression "$py_exe -m pip wheel -w Windows/wheeldir/ .."
popd
if ($LASTEXITCODE -ne 0) { throw "pip wheel failed." }

# create virtualenv for testing and install esig wheel into it.
Invoke-Expression "$py_exe -m virtualenv venv"
$wheel=(ls wheeldir\*.whl | Select-Object -First 1).Name

echo $wheel

ls .\venv\Scripts
.\venv\Scripts\python.exe -m pip install wheeldir\$wheel

# run tests
# TODO: Python 3.8+ doesn't use PATH to find dependent DLLs

echo (Invoke-Expression ".\venv\Scripts\python.exe --version")
.\venv\Scripts\python.exe -v -m unittest discover -v -s ..\..\esig\tests

if ($LASTEXITCODE -ne 0) {
   throw "Tests failed - will not copy wheel to output"
} else {
   echo "Tests passed - copying wheel to output"
   mkdir output
   mv wheeldir/esig*.whl output/
}
