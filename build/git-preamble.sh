#!/bin/bash    # not using -e here as the auth_header assignment can fail silently
# $1: GitHub password or Personal Access Token for rolyp
set -u -o xtrace

if [ ! -z "${RECOMBINE_USER}" -a ! -z "${RECOMBINE_TOKEN}"} ]; then
  git config --global url."https://github.com/".insteadOf "git@github.com:"
  auth_header="$(git config --local --get http.https://github.com/.extraheader)"
  git submodule sync --recursive
  git -c "http.extraheader=$auth_header" -c protocol.version=2 submodule update --init --force --recursive --depth=1

  git clone https://rolyp:$1@github.com/terrylyons/recombine.git build/recombine
fi