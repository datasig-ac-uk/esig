FROM microsoft/windowsservercore
## from the folder with this file and the context execute
## docker build -t windowscore -f Dockerfile.dockerfile .

# download and install visual studio
RUN powershell "Invoke-WebRequest https://download.microsoft.com/download/7/9/6/796EF2E4-801B-4FC4-AB28-B59FBF6D907B/VCForPython27.msi -OutFile .\VCForPython27.msi"
RUN powershell "msiexec.exe /i /quiet VCForPython27.msi ALLUSERS=1"
# Download boost zip file, unpack to a folder called "boost", and from the boost_144/boost 
# folder, install the boost libraries.   Set an env variable BOOST_ROOT pointing to this boost folder.

RUN powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; Invoke-WebRequest -UseBasicParsing http://sourceforge.net/projects/boost/files/boost/1.44.0/boost_1_44_0.zip -OutFile boost_144_0.zip"
RUN powershell Expand-Archive .\boost_144_0.zip -DestinationPath boost
RUN powershell Set-Item -path env:BOOST_ROOT -value 'boost\boost_1_44_0
# RUN setx /m BOOST_ROOT %CD%
# download and install 64-bit and 32-bit libraries for boost 1.44
RUN powershell "Invoke-WebRequest http://boost.teeks99.com/bin/boost_1_44_0-vc64-bin.exe -OutFile boost_1_44_0-vc64-bin.exe"
RUN powershell "Invoke-WebRequest http://boost.teeks99.com/bin/boost_1_44_0-vc32-bin.exe -OutFile boost_1_44_0-vc32-bin.exe"
RUN powershell "cmd.exe boost_1_44_0-vc64-bin.exe /quiet"
RUN powershell "cmd.exe boost_1_44_0-vc32-bin.exe /quiet"
# download and install python 2.7 and (later 3.4, 3.5, 3.6) (64 bit and 32 bit)
RUN powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; Invoke-WebRequest https://www.python.org/ftp/python/2.7.15/python-2.7.15.msi -OutFile python-2.7.15.msi"
RUN powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; Invoke-WebRequest https://www.python.org/ftp/python/3.6.5/python-3.6.5.exe -OutFile python-3.6.5.exe"
RUN powershell "msiexec.exe /quiet /i python-2.7.15.msi"
RUN powershell "cmd.exe /quiet python-3.6.5.exe"
ENTRYPOINT ["powershell"]