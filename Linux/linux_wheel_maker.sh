#!/bin/bash

### this is the command to be run in the docker container to build the wheels
### from the gzip-ed source.

for gz in $(ls esig*.gz); do ver=${gz%%.tar*}; for py in $(ls /opt/python); do pyexe=/opt/python/$py/bin/python && $pyexe -m pip install -U pip wheel virtualenv && $pyexe -m pip wheel $gz && auditwheel repair $ver-$py-linux_x86_64.whl && $pyexe -m virtualenv /tmp/$py && . /tmp/$py/bin/activate && pip install wheelhouse/$ver-$py-manylinux1_x86_64.whl && python -c 'import esig.tests as tests; tests.run_tests()' && deactivate && rm -rf /tmp/$py/ ; done ; done
