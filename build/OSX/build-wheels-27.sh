#!/bin/bash
# See build-wheels.sh for documentation.

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
