#!/bin/bash -e

# Each of these runs as a parallel job in the GitHub runner.
. build-wheel-27.sh
. build-wheel-34.sh
. build-wheel-other.sh 3.5.5
. build-wheel-other.sh 3.6.5
. build-wheel-other.sh 3.7.0
. build-wheel-other.sh 3.8.1
