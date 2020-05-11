#!/bin/bash -e
# Publish to PyPI. Publish step will fail unless version number is bumped (currently a manual step).
set -u -o xtrace

ls -la dist/
python -m pip install --upgrade twine==3.1.1
python -m twine upload -u __token__ -p {{ secrets.testpypi_password }} -r testpypi dist/
echo Uploaded to PyPI.
