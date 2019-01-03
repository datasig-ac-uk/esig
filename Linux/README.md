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

### Build the Docker images

You only need to do this once, unless you need to use a newer version of boost or another dependency at a
later date.
```
docker build -t manylinux_i686_boost -f Dockerfile_i686.dockerfile .
docker build -t manylinux_x86_64_boost -f Dockerfile_x86_64.dockerfile .
```

## Build the esig wheels 
### From Linux or OSX

```
for arch in i686 x86_64; do docker run --rm -v ${PWD}:/data manylinux_${arch}_boost "source ~/.bashrc ; cd /data; source linux_wheel_maker.sh $arch"; done;
```

### From Windows
```
docker run --rm -v "%CD%":/data manylinux1_x86_64_boost "source ~/.bashrc ; cd /data; for gz in $(ls esig*.gz); do ver=${gz%%.tar*}; for py in $(ls /opt/python); do pyexe=/opt/python/$py/bin/python && $pyexe -m pip install -U pip wheel virtualenv && $pyexe -m pip wheel $gz && auditwheel repair $ver-$py-linux_i686.whl && $pyexe -m virtualenv /tmp/$py && . /tmp/$py/bin/activate && pip install wheelhouse/$ver-$py-manylinux1_i686.whl && python -c 'import esig.tests as tests; tests.run_tests(True)' && deactivate && rm -rf /tmp/$py/ ; done ; done"

docker run --rm -v "%CD%":/data manylinux1_i686_boost "source ~/.bashrc ; cd /data; for gz in $(ls esig*.gz); do ver=${gz%%.tar*}; for py in $(ls /opt/python); do pyexe=/opt/python/$py/bin/python && $pyexe -m pip install -U pip wheel virtualenv && $pyexe -m pip wheel $gz && auditwheel repair $ver-$py-linux_i686.whl && $pyexe -m virtualenv /tmp/$py && . /tmp/$py/bin/activate && pip install wheelhouse/$ver-$py-manylinux1_i686.whl && python -c 'import esig.tests as tests; tests.run_tests(True)' && deactivate && rm -rf /tmp/$py/ ; done ; done"
```

The esig wheel files for the different python versions should now be in the ```wheelhouse/``` directory.
