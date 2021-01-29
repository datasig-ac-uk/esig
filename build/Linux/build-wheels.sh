#!/bin/bash -e

# Each of these runs as a parallel job in the GitHub runner.
./build-wheel.sh cp35-cp35m x86_64
./build-wheel.sh cp36-cp36m x86_64
./build-wheel.sh cp37-cp37m x86_64
./build-wheel.sh cp38-cp38 x86_64
#./build-wheel.sh cp35-cp35m i686
#./build-wheel.sh cp36-cp36m i686
#./build-wheel.sh cp37-cp37m i686
#./build-wheel.sh cp38-cp38 i686
