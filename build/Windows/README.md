## Building esig for Windows

### Prerequisites - docker-for-windows

Instructions for installing docker-for-windows can be found [here](https://www.docker.com/docker-windows)

### Build or pull the docker image

To build the docker image, run the command
```
docker build -t esig_builder_windows -f Dockerfile.dockerfile .
```
This will take a long time, and will result in a big (>70GB) docker image.  

You can alternatively pull a pre-built image from Dockerhub with:
```
docker pull nbarlow/esig_builder_windows:latest
```
(Again, note that this image is large, so ensure you have sufficient disk space).  If you do this, replace ```esig_builder_windows``` with ```nbarlow/esig_builder_windows:latest``` in the ```build_all_versions.bat``` batch script.

### Get esig source

Get the ```esig``` source code as a ```.tar.gz``` file.  You can download
the latest released version from the [PyPi downloads page](https://pypi.org/project/esig/#files)
and put it in this directory.

### Supported python versions

The list of targets (python version, and 32- or 64-bit) is in the text file ```python_versions.txt```.
Currently the list of versions is: 2.7, 3.5, 3.6, 3.7, all in both 32- and 64-bit.

### Build esig wheels in the docker container

Run the following command from Windows Command Prompt, from this directory:

```
.\build_all_versions.bat <esig_tar_gz_filename>
```
This will loop through all the python versions in ```python_versions.txt``` and for each one run issue a ```docker run``` command that will:
* Set the PATH environment variable to point to the requested python version.
* Build the ```esig``` wheel
* Setup a clean ```virtualenv``` environment for testing.
* Run the esig test suite.

If, and only if, all tests pass, the esig wheels will be copied to the ```output``` directory.
