#!/bin/bash
# See build-wheels.sh for documentation.

# MacPorts (for Python 3.4)
macos=$(sw_vers -productVersion)
macos_short=${macos%.*}

if [ $macos_short == "10.14" ]
then
   macPorts="MacPorts-2.6.2-10.14-Mojave.pkg"
elif [ $macos_short == "10.15" ]
then
   macPorts="MacPorts-2.6.2-10.15-Catalina.pkg"
else
   echo "Need to specify MacPorts download for ${macos_short}."
   exit 1
fi

curl -O https://distfiles.macports.org/MacPorts/$macPorts
sudo installer -verbose -pkg $macPorts -target /
rm $macPorts

if [ $? -eq 0 ]
then
   echo "MacPorts installed successfully."
   sudo /opt/local/bin/port select --set python3 python34
   which python3
else
   echo "MacPorts installation failed."
   exit 1
fi

# For 3.4
sudo /opt/local/bin/port -N install python34
sudo /opt/local/bin/port -N install py34-pip
sudo /opt/local/bin/port -N install boost

# For 2.7
brew install boost
brew install openssl # required now?

# TODO: factor out common bit.
pyexe=/usr/local/bin/python  # Python 2.7 (preinstalled)
py=$($pyexe --version 2>&1)
p=${py##* }
if [ $p == "2.7.17" ]
then
   . mac_wheel_builder-27-34.sh p
else
   echo "Expecting Python $p"
fi

pyexe=$(which python3 2>&1)  # Python 3.4 (MacPorts)
py=$($pyexe --version 2>&1)
p=${py##* }
if [ $p == "3.4.10" ]
then
   . mac_wheel_builder-27-34.sh p
else
   echo "Expecting Python $p"
fi
