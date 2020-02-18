## Building esig for OSX

It is not straightforward (and arguably not legal) to run OSX in a Docker
container, so the procedure here is run directly on a Mac, using
Python virtualenv environments.

### To build on OSX

### Prerequisites

This process assumes that `brew` is installed.

### Python versions

The ```esig``` package is currently built for the python versions listed in
```python_versions.txt```.  This file can be modified as necessary, with
one python version number per line.
To install all these python versions via pyenv, you can use the script
```
install_all_python_versions.sh
```

### Building the esig wheels

Assuming you have run the script above to install all the python versions
listed in ```python_versions.py```, you can just run the script
```
mac_wheel_builder.sh
```
and it will loop through all the python versions, and for each one:
 * create and activate a
virtual environment
 * install python dependencies
 * compile the python "wheel"
 * "delocate" the wheel, i.e. combine other dependencies, to give a portable binary.
 * create another virtual environment and install the newly created wheel
 * run the esig unit tests.
 
*If* the tests pass, output wheels will be in the ```output/``` subdirectory.
