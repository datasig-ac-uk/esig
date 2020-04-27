param([string] $vs_version,            # {9.0, 14.1}
      [string] $arch,                  # {32, 64}
      [string] $py_install_dir,
      [string] $py_installer,
      [string] $boost_platform_dir,
      [string] $boost_lib_dir,
      [string] $boost_installer)

Set-PSDebug -Trace 1

..\git-preamble.sh

if ($vs_version -eq "14.1") {
   # Use pre-installed Visual Studio
} elseif ($vs_version -eq "9.0") {
   curl -O https://download.microsoft.com/download/7/9/6/796EF2E4-801B-4FC4-AB28-B59FBF6D907B/VCForPython27.msi
   msiexec.exe /i VCForPython27.msi /quiet
} else {
   exit 1
}

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

# self-extracting installers - will unpack to C:\local\boost\boost_1_68_0\lib[64,32]-msvc-[version]
# without /VERYSILENT installer will attempt to open dialog box and silently fail
Start-Process -Wait -PassThru -FilePath .\$boost_installer -ArgumentList '/VERYSILENT /SP-'

mkdir $ENV:BOOST_ROOT\$boost_platform_dir
mkdir $ENV:BOOST_ROOT\$boost_platform_dir\lib

Move-Item -Path C:\local\boost_1_68_0\$boost_lib_dir\*.lib -Destination $ENV:BOOST_ROOT\$boost_platform_dir\lib

# TODO: combine these into one.
if ([System.IO.Path]::GetExtension($py_installer) -eq ".exe") {
   curl -L -o install-python.exe https://www.python.org/ftp/python/$py_installer
   $ErrorActionPreference = 'Stop'
   $VerbosePreference = 'Continue'
   Start-Process -Wait -PassThru -FilePath .\install-python.exe -ArgumentList '/quiet'
}
elseif ([System.IO.Path]::GetExtension($py_installer) -eq ".msi") {
   curl -L -o install-python.msi https://www.python.org/ftp/python/$py_installer
   Start-Process -Wait -PassThru -FilePath .\install-python.msi -ArgumentList '/quiet'
   echo $LASTEXITCODE
} else {
   exit 1
}

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
   Invoke-Expression "$py_exe -m pip wheel -b Windows/ -w Windows/output/ .."
popd
if ($LASTEXITCODE -ne 0) { throw "pip wheel failed." }

# create virtualenv for testing and install esig wheel into it.
Invoke-Expression "$py_exe -m virtualenv venv"
$wheel=(ls output\*.whl | Select-Object -First 1).Name
echo $wheel
ls .\venv
.\venv\Scripts\python.exe -m pip install output\$wheel
# run tests
.\venv\Scripts\\python.exe -c "import esig.tests as tests; tests.run_tests(terminate=True)"
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
