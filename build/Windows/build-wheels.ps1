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

# Would like to get esig source as .tar.gz. We can't build from sources as Python not installed in 
# Windows build environment. Instead just assume in build directory for now.

# TODO: fix script that is run.
# TODO: migrate all behaviour from build_all_versions.bat and then delete that file.
docker run --rm -v ${PWD}:C:\data esig_builder_windows "$env:PATH = Get-Content -Path pathenv_python35_32; cd data; .\windows_wheel_maker.bat esig-0.6.31.tar.gz $p"
