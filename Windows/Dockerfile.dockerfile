FROM microsoft/windowsservercore as esig_build_base
## from the folder with this file and the context execute
## docker build -t windows_p27_64 -f Dockerfile_p27_64.dockerfile .

# install chocolatey package manager and use it to get GNU wget
RUN powershell "Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
RUN powershell "choco install -y wget"

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

###### the preceding should all be independent of python version, so the final images can all be built on top of that layer
###### first lets build the image for python 2.7

FROM esig_build_base as esig_build_python27

# download and install visual studio C++ compiler
RUN powershell "Invoke-WebRequest https://download.microsoft.com/download/7/9/6/796EF2E4-801B-4FC4-AB28-B59FBF6D907B/VCForPython27.msi -OutFile .\VCForPython27.msi"
RUN msiexec.exe /i VCForPython27.msi /quiet

# download and install python 2.7 and (later 3.4, 3.5, 3.6) (64 bit and 32 bit)
RUN powershell "choco install -y python --version 2.7.10"
### following env variable lets us use python 2.7 in the docker container..
ENV PYTHONIOENCODING 'UTF-8'

#### choco install pip should give us pip.exe and pip3.exe in tools\python\Scripts
RUN powershell "choco install -y  --allow-empty-checksums pip"
RUN powershell "pip.exe install wheel"
RUN powershell "pip.exe install numpy"
RUN powershell "pip.exe install delocate"

ENTRYPOINT ["powershell"]

####### now the image for python 3.7

FROM esig_build_base as esig_build_python37
#RUN powershell "wget.exe --no-check-certificate https://download.microsoft.com/download/D/1/4/D142F7E7-4D7E-4F3B-A399-5BACA91EB569/vs_Community.exe"
RUN powershell "wget.exe --no-check-certificate https://download.visualstudio.microsoft.com/download/pr/5426f054-a10a-441f-b8a9-f7135d58d59b/48510132eb9254121dd72072197308f6/vs_buildtools.exe"
RUN .\vs_buildtools.exe  --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --includeOptional --quiet
RUN powershell ".\vs_buildtools.exe  --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --includeOptional --quiet"
#RUN .\vs_Community.exe --add Microsoft.VisualStudio.Workload.VCTools --quiet
RUN powershell "choco.exe install vswhere"
RUN powershell "choco install -y python3"
RUN powershell "pip.exe install wheel"
RUN powershell "pip.exe install numpy"
RUN powershell "pip.exe install delocate"
ENTRYPOINT ["powershell"]
