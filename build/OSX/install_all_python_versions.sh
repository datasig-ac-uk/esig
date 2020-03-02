#!/bin/bash
# simple script to install several python versions using pyenv.

eval "$(pyenv init -)"

for p in $( cat python_versions.txt ); do
   # see https://github.com/pyenv/pyenv/wiki/common-build-problems
   # CFLAGS="-I$(brew --prefix openssl)/include" \
   # LDFLAGS="-L$(brew --prefix openssl)/lib" \
   # CFLAGS="-I./libressl-2.2.7/include" \ 
   # CPPFLAGS="-I./libressl-2.2.7/include" \
   # LDFLAGS="-L./libressl-2.2.7/lib" \
   pyenv install --skip-existing $p
done
