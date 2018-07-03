#!/bin/bash

# simple script to install several python versions using pyenv.
# pyenv can be installed using homebrew:
# brew install pyenv

eval "$(pyenv init -)"

for p in $( cat python_versions.txt ); do
	 pyenv install $p
done
