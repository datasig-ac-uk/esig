## Building esig for Windows

### Note - currently the recipe works for Python 2.7 / 64-bit.
### It should be straightforward to substitute different python/boost/VisualStudio compiler versions in, but we need to
### investigate whether we need separate dockerfiles for each, or whether multiple versions can be done in the same docker container.


### Prerequisites - docker-for-windows

Instructions for installing docker-for-windows can be found [here](https://www.docker.com/docker-windows)

### Building the docker image

Run the command
```
docker build -t windows_p27_64 -f Dockerfile_p27_64.dockerfile .
```

### Get esig source

Get the ```esig``` source code as a ```.tar.gz``` file.  You can download
the latest released version from the [PyPi downloads page](https://pypi.org/project/esig/#files)
and put it in this directory.


### Build esig wheel in docker container

Run the following command from Windows Command Prompt:
```
docker run --rm -v "%CD%":C:\data windows_p27_64 "cd data; pip wheel --no-binary -b latest <esig_tar.gz_filename>"
```
or from Powershell, do:
```
docker run --rm -v "$pwd":C:\data windows_p27_64 "cd data; pip wheel --no-binary -b latest <esig_tar.gz_filename>"
```
and you should get the esig wheel in your current directory.