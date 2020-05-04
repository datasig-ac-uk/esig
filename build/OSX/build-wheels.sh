#!/bin/bash -e

# Each of these runs as a parallel job in the GitHub runner.
. build-wheels-27.sh
. build-wheels-34.sh
. build-wheels-rest.sh
