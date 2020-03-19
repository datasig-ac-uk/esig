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
else
   echo "MacPorts installation failed."
   exit 1
fi

# For 3.4
# Installing python34 won't always repair a broken installation; uninstall first for reproducibility.
sudo /opt/local/bin/port -N uninstall --follow-dependents python34
sudo /opt/local/bin/port -N install python34
sudo /opt/local/bin/port select --set python3 python34 # perhaps not needed but sanity-checks python34 ok
if [ $? -ne 0 ]
then
   echo "Couldn't find Python 3.4 after MacPorts installation."
   exit 1
fi
sudo /opt/local/bin/port -N install py34-pip
sudo /opt/local/bin/port -N install boost

# For 2.7
brew install boost
brew install openssl # required now?

# TODO: relax version checks below to major.minor.

# Python 2.7 (preinstalled)
pyexe=/usr/local/bin/python
py=$($pyexe --version 2>&1)
p=${py##* }
expect="2.7.17"
if [ $p == $expect ]
then
   . mac_wheel_builder-27-34.sh p ""
else
   echo "Expecting Python $expect, got $p"
   exit 1
fi

# Python 3.4 (MacPorts)
pyexe=/opt/local/bin/python3.4 # MacPorts seems to consistently install here
py=$($pyexe --version 2>&1)
p=${py##* }
expect="3.4.10"
if [ $p == $expect ]
then
   . mac_wheel_builder-27-34.sh p "sudo"
else
   echo "Expecting Python $expect, got $p"
   exit 1
fi
