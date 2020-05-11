#!/bin/bash -e
# Publish to PyPI. Publish step will fail unless version number is bumped (currently a manual step).
# $1: PyPI password
# $2: PyPI repository to use (pypi or testpypi)
set -u -o xtrace

ls -la dist/
python -m pip install --upgrade twine==3.1.1
python -m twine upload -u __token__ -p "$1" -r "$2" dist/*
echo Uploaded to PyPI.
