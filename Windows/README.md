## Building esig for Windows

### Note - currently the recipe works for Python 2.7, 3.5, 3.6, 3.7, 64-bit and 32-bit.  Python 3.4 support is work-in-progress.


### Prerequisites - docker-for-windows

Instructions for installing docker-for-windows can be found [here](https://www.docker.com/docker-windows)

### Building the docker image

Run the command
```
docker build -t esig_builder_windows -f Dockerfile.dockerfile .
```
This will take a long time, and will result in a big docker image.  You can alternatively pull a pre-built image from Dockerhub with:
```
docker pull nbarlow/esig_builder_windows:latest
```


### Get esig source

Get the ```esig``` source code as a ```.tar.gz``` file.  You can download
the latest released version from the [PyPi downloads page](https://pypi.org/project/esig/#files)
and put it in this directory.


### Build esig wheel in docker container

Run the following command from Windows Command Prompt:

```
docker run --rm -v "%CD%":C:\data esig_builder_windows "$env:PATH = Get-Content -Path pathenv_<PYTHON_VERSION>;cd data; python.exe -m pip wheel --no-binary -b latest <esig_tar.gz_filename>"
```
where PYTHON_VERSION is one of:
```
python37_64
python37_32
python36_64
python36_32
python35_64
python35_32
python27_64
python27_32
```

It should take a few minutes to build, and then you should get the
output wheel in your current working directory.
