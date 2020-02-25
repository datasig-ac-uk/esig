# Top-level script for building esig for Windows.
# Python wheels will be created for Python versions specified in python_versions.txt.

# Build docker image.
# This will take a long time, and will result in a big (>70GB) docker image.  
# You can alternatively pull a pre-built image from Dockerhub with:
# 
#   docker pull nbarlow/esig_builder_windows:latest
#
# (Again, note that this image is large, so ensure you have sufficient disk space). If you do this, 
# replace `esig_builder_windows` with `nbarlow/esig_builder_windows:latest` in the `build_all_versions.bat` 
# batch script.
docker build -t esig_builder_windows -f Dockerfile.dockerfile .

# Get esig source.
../sdist.ps1
