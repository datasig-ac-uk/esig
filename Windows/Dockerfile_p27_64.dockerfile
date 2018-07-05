FROM microsoft/windowsservercore
## from the folder with this file and the context execute
## docker build -t windows_p27_64 -f Dockerfile_p27_64.dockerfile .

# install chocolatey package manager and use it to get GNU wget
RUN powershell "Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
RUN powershell "choco install -y wget"

# download and install visual studio C++ compiler
RUN powershell "Invoke-WebRequest https://download.microsoft.com/download/7/9/6/796EF2E4-801B-4FC4-AB28-B59FBF6D907B/VCForPython27.msi -OutFile .\VCForPython27.msi"
RUN msiexec.exe /i VCForPython27.msi /quiet

# Download boost zip file, unpack to a folder called "boost".
RUN wget.exe --no-check-certificate http://sourceforge.net/projects/boost/files/boost/1.44.0/boost_1_44_0.zip
RUN powershell Expand-Archive .\boost_1_44_0.zip -DestinationPath boost

# set BOOST_ROOT env variable to point to this boost directory
ENV BOOST_ROOT 'C:\boost\boost_1_44_0'

# download pre-built 64-bit and 32-bit libraries for boost 1.44
RUN powershell "Invoke-WebRequest http://boost.teeks99.com/bin/boost_1_44_0-vc64-bin.exe -OutFile boost_1_44_0-vc64-bin.exe"
RUN powershell "Invoke-WebRequest http://boost.teeks99.com/bin/boost_1_44_0-vc32-bin.exe -OutFile boost_1_44_0-vc32-bin.exe"
# these should be self-extracting - just execute the commands and the libs will be unpacked into lib64 (lib32)
RUN powershell ".\boost_1_44_0-vc64-bin.exe"
RUN powershell ".\boost_1_44_0-vc32-bin.exe"
# now copy those directories to where the compiler expects them (based on BOOST_ROOT env var)
RUN powershell "mkdir boost\boost_1_44_0\x64"
RUN powershell "Move-Item lib64 boost\boost_1_44_0\x64\lib"

# download and install python 2.7 and (later 3.4, 3.5, 3.6) (64 bit and 32 bit)
RUN powershell "choco install -y python --version 2.7.10"
### following env variable lets us use python 2.7 in the docker container..
ENV PYTHONIOENCODING 'UTF-8'

#### choco install pip should give us pip.exe and pip3.exe in tools\python\Scripts
RUN powershell "choco install -y  --allow-empty-checksums pip"
RUN powershell "pip.exe install wheel"
RUN powershell "pip.exe install numpy"
RUN powershell "pip.exe install delocate"
