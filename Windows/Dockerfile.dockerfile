FROM microsoft/dotnet-framework:4.7.1 as esig_build_base_boost168

## from the folder with this file and the context execute
## docker build -t esig_builder_windows -f Dockerfile_combine.dockerfile .

# install chocolatey package manager and use it to get GNU wget
RUN powershell "Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
RUN powershell "choco install -y wget"

######### BOOST #################################



# Download boost 1.68 zip file containing the headers, unpack to a folder called "boost".

RUN wget.exe --no-check-certificate https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.zip
RUN powershell Expand-Archive .\boost_1_68_0.zip -DestinationPath boost

# set BOOST_ROOT env variable to point to this boost directory
ENV BOOST_ROOT 'C:\boost\boost_1_68_0'

# download pre-built 64-bit and 32-bit libraries for boost 1.68

RUN powershell "wget.exe --no-check-certificate https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-9.0-64.exe/download -O boost_1_68_0-msvc-9.0-64.exe"
RUN powershell "wget.exe --no-check-certificate https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-9.0-32.exe/download -O boost_1_68_0-msvc-9.0-32.exe"
RUN powershell "wget.exe --no-check-certificate https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-10.0-64.exe/download -O boost_1_68_0-msvc-10.0-64.exe"
RUN powershell "wget.exe --no-check-certificate https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-10.0-32.exe/download -O boost_1_68_0-msvc-10.0-32.exe"
RUN powershell "wget.exe --no-check-certificate https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-14.0-64.exe/download -O boost_1_68_0-msvc-14.0-64.exe"
RUN powershell "wget.exe --no-check-certificate https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-14.0-32.exe/download -O boost_1_68_0-msvc-14.0-32.exe"
RUN powershell "wget.exe --no-check-certificate https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-14.1-64.exe/download -O boost_1_68_0-msvc-14.1-64.exe"
RUN powershell "wget.exe --no-check-certificate https://sourceforge.net/projects/boost/files/boost-binaries/1.68.0/boost_1_68_0-msvc-14.1-32.exe/download -O boost_1_68_0-msvc-14.1-32.exe"


# these should be self-extracting - just execute the command and the libs will be unpacked into C:\local\boost\boost_1_68_0\lib[64,32]-msvc-[version]
RUN .\boost_1_68_0-msvc-9.0-64.exe /VERYSILENT /SP-
RUN .\boost_1_68_0-msvc-9.0-32.exe /VERYSILENT /SP-
RUN .\boost_1_68_0-msvc-10.0-64.exe /VERYSILENT /SP-
RUN .\boost_1_68_0-msvc-10.0-32.exe /VERYSILENT /SP-
RUN .\boost_1_68_0-msvc-14.0-64.exe /VERYSILENT /SP-
RUN .\boost_1_68_0-msvc-14.0-32.exe /VERYSILENT /SP-
RUN .\boost_1_68_0-msvc-14.1-64.exe /VERYSILENT /SP-
RUN .\boost_1_68_0-msvc-14.1-32.exe /VERYSILENT /SP-

# now copy those directories to where the compiler expects them (based on BOOST_ROOT env var)
RUN powershell "mkdir boost\boost_1_68_0\x64"
RUN powershell "mkdir boost\boost_1_68_0\x64\lib"

RUN powershell "mkdir boost\boost_1_68_0\x32"
RUN powershell "mkdir boost\boost_1_68_0\x32\lib"


## Copy boost libraries to correct location
RUN powershell "mkdir boost\boost_1_68_0\win32"
RUN powershell "mkdir boost\boost_1_68_0\win32\lib"
## "move" (or powershell equivalent) doesn't work! Just deletes files!
RUN copy .\local\boost_1_68_0\lib64-msvc-14.0\*.lib .\boost\boost_1_68_0\x64\lib
RUN copy .\local\boost_1_68_0\lib32-msvc-14.0\*.lib .\boost\boost_1_68_0\win32\lib
RUN copy .\local\boost_1_68_0\lib64-msvc-14.1\*.lib .\boost\boost_1_68_0\x64\lib
RUN copy .\local\boost_1_68_0\lib32-msvc-14.1\*.lib .\boost\boost_1_68_0\win32\lib
RUN copy .\local\boost_1_68_0\lib64-msvc-9.0\*.lib .\boost\boost_1_68_0\x64\lib
RUN copy .\local\boost_1_68_0\lib32-msvc-9.0\*.lib .\boost\boost_1_68_0\win32\lib
RUN copy .\local\boost_1_68_0\lib64-msvc-10.0\*.lib .\boost\boost_1_68_0\x64\lib
RUN copy .\local\boost_1_68_0\lib32-msvc-10.0\*.lib .\boost\boost_1_68_0\win32\lib
######### VISUAL STUDIO ##############################

## download and install visual studio C++ compiler for python 2.7
RUN powershell "Invoke-WebRequest https://download.microsoft.com/download/7/9/6/796EF2E4-801B-4FC4-AB28-B59FBF6D907B/VCForPython27.msi -OutFile .\VCForPython27.msi"
RUN msiexec.exe /i VCForPython27.msi /quiet

## Visual studio 10 for python 3.4

#RUN powershell "wget.exe --no-check-certificate https://download.microsoft.com/download/5/B/C/5BC5DBB3-652D-4DCE-B14A-475AB85EEF6E/vcredist_x86.exe"
#RUN powershell $ErrorActionPreference = 'Stop'; \
#      $VerbosePreference = 'Continue'; \
#      $p = Start-Process -Wait -PassThru -FilePath C:\vcredist_x86.exe -ArgumentList '/q';

## visual studio 14 for python 3.6+
RUN powershell "wget.exe --no-check-certificate https://download.visualstudio.microsoft.com/download/pr/5426f054-a10a-441f-b8a9-f7135d58d59b/48510132eb9254121dd72072197308f6/vs_buildtools.exe"

RUN powershell $ErrorActionPreference = 'Stop'; \
      $VerbosePreference = 'Continue'; \
      $p = Start-Process -Wait -PassThru -FilePath C:\vs_buildtools.exe -ArgumentList '--add Microsoft.VisualStudio.Workload.VCTools --includeOptional --includeRecommended --quiet --nocache --wait';


## copy rc.exe and rcdll.dll needed for VC14 to compile for python 3.5
RUN copy "C:\Program Files (x86)\Windows Kits\8.1\bin\x86\rc.exe" "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\"
RUN copy "C:\Program Files (x86)\Windows Kits\8.1\bin\x86\rcdll.dll" "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\"

####### PYTHON #########################################

## Will modify the PATH several times, so store the original one here so that we can go back to it inbetween doing different python versions
ENV ORIG_PATH="C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\ProgramData\chocolatey\bin;C:\Users\ContainerAdministrator\AppData\Local\Microsoft\WindowsApps"

## python 3.5 32-bit
RUN wget.exe --no-check-certificate https://www.python.org/ftp/python/3.5.4/python-3.5.4.exe
RUN powershell $ErrorActionPreference = 'Stop'; \
      $VerbosePreference = 'Continue'; \
      $p = Start-Process -Wait -PassThru -FilePath C:\python-3.5.4.exe -ArgumentList '/quiet';


ENV PATH="C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python35-32;${ORIG_PATH}"
RUN echo %PATH%>pathenv_python35_32
RUN python.exe -m pip install numpy
RUN python.exe -m pip install wheel
RUN python.exe -m pip install delocate
RUN python.exe -m pip install --upgrade setuptools


## python 3.5 64-bit
RUN wget.exe --no-check-certificate https://www.python.org/ftp/python/3.5.4/python-3.5.4-amd64.exe
RUN powershell $ErrorActionPreference = 'Stop'; \
      $VerbosePreference = 'Continue'; \
      $p = Start-Process -Wait -PassThru -FilePath C:\python-3.5.4-amd64.exe -ArgumentList '/quiet';

ENV PATH="C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python35;${ORIG_PATH}"
RUN echo %PATH%>pathenv_python35_64
RUN python.exe -m pip install numpy
RUN python.exe -m pip install wheel
RUN python.exe -m pip install delocate

## python 3.6 32-bit
#RUN wget.exe --no-check-certificate https://www.python.org/ftp/python/3.6.6/python-3.6.6.exe
#RUN powershell $ErrorActionPreference = 'Stop'; \
#      $VerbosePreference = 'Continue'; \
#      $p = Start-Process -Wait -PassThru -FilePath C:\python-3.6.6.exe -ArgumentList '/quiet';

#ENV PATH="C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python36-32;${ORIG_PATH}"
#RUN echo %PATH%>pathenv_python36_32
#RUN python.exe -m pip install numpy
#RUN python.exe -m pip install wheel
#RUN python.exe -m pip install delocate

## python 3.6 64-bit
#RUN wget.exe --no-check-certificate https://www.python.org/ftp/python/3.6.6/python-3.6.6-amd64.exe
#RUN powershell $ErrorActionPreference = 'Stop'; \
#      $VerbosePreference = 'Continue'; \
#      $p = Start-Process -Wait -PassThru -FilePath C:\python-3.6.6-amd64.exe -ArgumentList '/quiet';

#ENV PATH="C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python36;${ORIG_PATH}"
#RUN echo %PATH%>pathenv_python36_64
#RUN python.exe -m pip install numpy
#RUN python.exe -m pip install wheel
#RUN python.exe -m pip install delocate

## python 3.7 32-bit
#RUN wget.exe --no-check-certificate https://www.python.org/ftp/python/3.7.0/python-3.7.0.exe
#RUN powershell $ErrorActionPreference = 'Stop'; \
#      $VerbosePreference = 'Continue'; \
#      $p = Start-Process -Wait -PassThru -FilePath C:\python-3.7.0.exe -ArgumentList '/quiet';

#ENV PATH="C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python37-32;${ORIG_PATH}"
#RUN echo %PATH%>pathenv_python37_32
#RUN python.exe -m pip install numpy
#RUN python.exe -m pip install wheel
#RUN python.exe -m pip install delocate

## python 3.7 64-bit
#RUN wget.exe --no-check-certificate https://www.python.org/ftp/python/3.7.0/python-3.7.0-amd64.exe
#RUN powershell $ErrorActionPreference = 'Stop'; \
#      $VerbosePreference = 'Continue'; \
#      $p = Start-Process -Wait -PassThru -FilePath C:\python-3.7.0-amd64.exe -ArgumentList '/quiet';

#ENV PATH="C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python37;${ORIG_PATH}"
#RUN echo %PATH%>pathenv_python37_64
#RUN python.exe -m pip install numpy
#RUN python.exe -m pip install wheel


## python 3.4 32-bit
RUN wget.exe --no-check-certificate https://www.python.org/ftp/python/3.4.4/python-3.4.4.msi
RUN .\python-3.4.4.msi /quiet
RUN move C:\Python34 C:\Python34-32
ENV PATH="C:\Python34-32;C:\Python34-32\Scripts;${ORIG_PATH}"
RUN echo %PATH%>pathenv_python34_32
RUN python.exe -m pip install numpy
RUN python.exe -m pip install wheel


## python 3.4 64-bit
RUN wget.exe --no-check-certificate https://www.python.org/ftp/python/3.4.4/python-3.4.4.amd64.msi
### INSTALLATION OF THIS FAILS WITH EXIT CODE 1603 (fatal error), OR WHEN DOING BY HAND IN DOCKER CONTAINER FAILS SILENTLY - WHY???
#RUN .\python-3.4.4.amd64.msi /quiet
#RUN move C:\Python34 C:\Python34-64
#ENV PATH="C:\Python34-64;C:\Python34-64\Scripts;${ORIG_PATH}"
#RUN echo %PATH%>pathenv_python34_64
#RUN python.exe -m pip install numpy
#RUN python.exe -m pip install wheel


## python 2.7 32-bit
RUN wget.exe --no-check-certificate  https://www.python.org/ftp/python/2.7.15/python-2.7.15.msi
RUN .\python-2.7.15.msi /quiet
RUN move C:\Python27 C:\Python27-32
ENV PYTHONIOENCODING 'UTF-8'
ENV PATH="C:\Python27-32;C:\Python27-32\Scripts;${ORIG_PATH}"
RUN echo %PATH%>pathenv_python27_32
RUN python.exe -m pip install numpy
RUN python.exe -m pip install wheel


## python 2.7 64-bit
RUN wget.exe --no-check-certificate  https://www.python.org/ftp/python/2.7.15/python-2.7.15.amd64.msi
RUN .\python-2.7.15.amd64.msi /quiet
RUN move C:\Python27 C:\Python27-64
ENV PYTHONIOENCODING 'UTF-8'
ENV PATH="C:\Python27-64;C:\Python27-64\Scripts;${ORIG_PATH}"
RUN echo %PATH%>pathenv_python27_64
RUN python.exe -m pip install numpy
RUN python.exe -m pip install wheel



## stuff for python 3.4 / VS 10
## 1) Windows SDK for Windows 7 and .NET Framework 4.  Won't install compilers, as it thinks wrong version of .NET is installed (but we can't install
## the right version because Windows Server 2016 already has a newer version).
## web installer doesn't support silent install, so need to get ISOs  (THEN - HOW DO WE EXTRACT FILES FROM ISO?????) https://support.microsoft.com/en-us/help/2498225/installing-windows-sdk-7-1-in-silent-mode
# x86 version
RUN wget.exe --no-check-certificate https://download.microsoft.com/download/F/1/0/F10113F5-B750-4969-A255-274341AC6BCE/GRMSDK_EN_DVD.iso
# x86_64
RUN wget.exe --no-check-certificate https://download.microsoft.com/download/F/1/0/F10113F5-B750-4969-A255-274341AC6BCE/GRMSDKX_EN_DVD.iso
## get 7zip to extract
RUN choco install -y 7zip
RUN 7z.exe x -y -oC:\winSDK7.1-32 .\GRMSDK_EN_DVD.iso
RUN powershell $ErrorActionPreference = 'Stop'; \
      $VerbosePreference = 'Continue'; \
      $p = Start-Process -Wait -PassThru -FilePath C:\winSDK7.1-32\setup.exe -ArgumentList '-q params:ADDLOCAL=ALL';

RUN 7z.exe x -y -oC:\winSDK7.1-64 .\GRMSDKX_EN_DVD.iso
RUN powershell $ErrorActionPreference = 'Stop'; \
      $VerbosePreference = 'Continue'; \
      $p = Start-Process -Wait -PassThru -FilePath C:\winSDK7.1-64\setup.exe -ArgumentList '-q params:ADDLOCAL=ALL';


## 2) Install Visual C++ 2010 Service Pack 1 Compiler Update for SDK 7
RUN wget.exe --no-check-certificate https://download.microsoft.com/download/7/5/0/75040801-126C-4591-BCE4-4CD1FD1499AA/VC-Compiler-KB2519277.exe
RUN powershell $ErrorActionPreference = 'Stop'; \
      $VerbosePreference = 'Continue'; \
      $p = Start-Process -Wait -PassThru -FilePath C:\VC-Compiler-KB2519277.exe -ArgumentList '/q';


## 3) Install Microsoft Visual C++ 2010 Redistributable
RUN wget.exe --no-check-certificate https://download.microsoft.com/download/5/B/C/5BC5DBB3-652D-4DCE-B14A-475AB85EEF6E/vcredist_x86.exe
RUN powershell $ErrorActionPreference = 'Stop'; \
      $VerbosePreference = 'Continue'; \
      $p = Start-Process -Wait -PassThru -FilePath C:\vcredist_x86.exe -ArgumentList '/q';
RUN wget.exe --no-check-certificate https://download.microsoft.com/download/3/2/2/3224B87F-CFA0-4E70-BDA3-3DE650EFEBA5/vcredist_x64.exe
RUN powershell $ErrorActionPreference = 'Stop'; \
      $VerbosePreference = 'Continue'; \
      $p = Start-Process -Wait -PassThru -FilePath C:\vcredist_x64.exe -ArgumentList '/q';


ENTRYPOINT ["powershell"]
