## Building esig for Linux

Python wheels will be created for 64-bit (x86_64)
and for 32-bit (i686), for python versions 2.7, 3.4, 3.5, 3.6 and 3.7. 

### Prerequisites: Docker

This process relies on having [Docker](https://docs.docker.com/)
installed and running.  There are minor differences in the docker commands to
run if you are using Windows (and docker-for-windows) or a posix system, but
the Dockerfiles themselves are identical.
To install docker (or docker-for-windows) on your system, follow the instructions
[here](https://docs.docker.com/).

### Get esig source

Get the ```esig``` source code as a ```.tar.gz``` file.  You can download
the latest released version from the [PyPi downloads page](https://pypi.org/project/esig/#files)
and put it in this directory.

### Build or pull the Docker images

You only need to do this once, unless you need to use a newer version of boost or another dependency at a
later date.
```
docker build -t esig_builder_linux_i686 -f Dockerfile_i686.dockerfile .
docker build -t esig_builder_linux_x86_64 -f Dockerfile_x86_64.dockerfile .
```
Alternatively, you can pull these from dockerhub:
```
docker pull nbarlow/esig_builder_linux_i686:latest
docker pull nbarlow/esig_builder_linux_x86_64:latest
```

## Build the esig wheels 

The ```linux_wheel_maker.sh``` script is run inside the docker container, and performs the steps to build the esig binary wheel, run tests, and if the tests are successful, copy the wheel to the ```wheelhouse``` directory.


### From Linux or OSX

```
for arch in i686 x86_64; do docker run --rm -v ${PWD}:/data esig_builder_linux_${arch} "source ~/.bashrc ; cd /data; source linux_wheel_maker.sh $arch"; done;
```

### From Windows

Ensure that docker for windows has "experimental features" enabled, in order to run linux containers.
```
docker run --platform=linux --rm -v "%CD%":/data esig_builder_linux_i686 "source ~/.bashrc ; cd /data; source linux_wheel_maker.sh i686; done;"
docker run --platform=linux --rm -v "%CD%":/data esig_builder_linux_x86_64 "source ~/.bashrc ; cd /data; source linux_wheel_maker.sh x86_64; done;"
```

The esig wheel files for the different python versions should now be in the ```wheelhouse/``` directory.
