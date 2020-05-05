#!/bin/bash -e
set -u -o xtrace
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

echo MacPorts installer requires sudo.
curl -O https://distfiles.macports.org/MacPorts/$macPorts
sudo installer -verbose -pkg $macPorts -target /
rm $macPorts

# Installing python34 won't always repair a broken installation; uninstall first for reproducibility.
sudo /opt/local/bin/port -N uninstall --follow-dependents python34
sudo /opt/local/bin/port -N install python34
sudo /opt/local/bin/port select --set python3 python34 # perhaps not needed but sanity-checks python34 ok
sudo /opt/local/bin/port -N install py34-pip
sudo /opt/local/bin/port -N install boost

# TODO: relax version checks below to major.minor.
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
