FROM microsoft/dotnet-framework:4.7.1 as esig_build_base_boost168

## from the folder with this file and the context execute
## docker build -t esig_build_python37 -f Dockerfile_p37.dockerfile .

## install chocolatey package manager and use it to get GNU wget
RUN powershell "Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
RUN powershell "choco install -y wget"

## Download boost zip file, unpack to a folder called "boost".
RUN wget.exe --no-check-certificate https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.zip
RUN powershell Expand-Archive .\boost_1_68_0.zip -DestinationPath boost

## set BOOST_ROOT env variable to point to this boost directory
ENV BOOST_ROOT 'C:\boost\boost_1_68_0'

## download pre-built 64-bit and 32-bit libraries for boost 1.68
RUN powershell "wget.exe --no-check-certificate https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-14.1-64.exe/download -O boost_1_68_0-msvc-14.1-64.exe"

## these should be self-extracting - just execute the command and the libs will be unpacked into C:\local\boost\boost_1_68_0\lib64-msvc-14.1
RUN .\boost_1_68_0-msvc-14.1-64.exe /VERYSILENT /SP-

## now copy those directories to where the compiler expects them (based on BOOST_ROOT env var)
RUN powershell "mkdir boost\boost_1_68_0\x64"
RUN powershell "Move-Item C:\local\boost_1_68_0\lib64-msvc-14.1 .\boost\boost_1_68_0\x64\lib"


## Download the installer for Visual Studio 2017 build tools
RUN powershell "wget.exe --no-check-certificate https://download.visualstudio.microsoft.com/download/pr/5426f054-a10a-441f-b8a9-f7135d58d59b/48510132eb9254121dd72072197308f6/vs_buildtools.exe"

## run the Visual Studio installer like this to ensure it is blocking (takes
## a long time).

RUN powershell $ErrorActionPreference = 'Stop'; \
      $VerbosePreference = 'Continue'; \
      $p = Start-Process -Wait -PassThru -FilePath C:\vs_buildtools.exe -ArgumentList '--add Microsoft.VisualStudio.Workload.VCTools --includeOptional --includeRecommended --quiet --nocache --wait';


####### now the image for python 3.7

FROM esig_build_base_boost168 as esig_build_python37
RUN powershell "choco.exe install -y vswhere"
RUN powershell "choco.exe install -y python3"

RUN powershell "pip.exe install wheel"
RUN powershell "pip.exe install numpy"
RUN powershell "pip.exe install delocate"

ENTRYPOINT ["powershell"]
