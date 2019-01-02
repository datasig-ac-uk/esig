#!/bin/bash

# simple script to install several python versions using pyenv.
# pyenv can be installed using homebrew:
# brew install pyenv

# openssl is also needed for python 3.7
# brew install openssl

eval "$(pyenv init -)"

for p in $( cat python_versions.txt ); do
          CFLAGS="-I$(brew --prefix openssl)/include" \
	  LDFLAGS="-L$(brew --prefix openssl)/lib" \
	  pyenv install $p
done
