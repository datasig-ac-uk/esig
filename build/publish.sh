#!/bin/bash -e

ls -la dist/
python -m pip install --upgrade twine==3.1.1
python -m twine upload -u __token__ -p {{ secrets.testpypi_password }} -r testpypi dist/*
git commit -a -m "Bump version number."
git push
git push --tags
echo Uploaded to PyPI.
