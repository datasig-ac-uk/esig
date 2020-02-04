#!/bin/bash

### this is the command to be run in the docker container to build the wheels
### from the gzip-ed source.

arch=$1

# TODO: what exactly are the semicolons here for?
# TODO: align do/done delimiters

for gz in $(ls esig*.gz);
 do ver=${gz%%.tar*};
 for py in $(ls /opt/python);
    do pyexe=/opt/python/$py/bin/python
	 $pyexe -m pip install -U pip virtualenv
	 $pyexe -m pip install wheel==0.31.1
	 $pyexe -m pip install -U numpy
	 $pyexe -m pip wheel $gz
	 auditwheel repair $ver-$py-linux_$arch.whl
	 $pyexe -m virtualenv /tmp/$py
	 . /tmp/$py/bin/activate
	 pip install wheelhouse/$ver-$py-manylinux1_$arch.whl
	 python -c 'import esig.tests as tests; tests.run_tests(terminate=True)'
	 if [ $? -eq 0 ]
	 then
	     echo "Tests passed"
	 else
	     echo "Tests failed - removing wheel"
             rm wheelhouse/$ver-$py-manylinux1_$arch.whl
	 fi
	 deactivate
	 rm -rf /tmp/$py/ ;
 done ;
done
## now remove all the wheels from the current directory -
## (the final ones we want will be in "wheelhouse")
rm *.whl
