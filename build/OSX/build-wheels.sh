#!/bin/bash
# Top-level build script.

brew install boost
brew install pyenv
brew install pyenv-virtualenv
brew install openssl

# How persistent is this URL?
wget --no-clobber https://files.pythonhosted.org/packages/7c/4f/d0dd6fd1054110efc73ae745036fb57e37017ed45ed449bf472f92197024/esig-0.6.31.tar.gz
. install_all_python_versions.sh
. mac_wheel_builder.sh
