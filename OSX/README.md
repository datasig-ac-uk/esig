## Building esig for OSX.

It is not straightforward (and arguably not legal) to run OSX in a Docker
container, so the procedure here is run directly on a Mac machine, using
python virtualenv environments.

### Prerequisites

It is necessary to have ```boost``` installed in order to build esig.
For the script below, in order to build for many python versions, ```pyenv```
and ```pyenv-virtualenv``` are used.  All of these can be installed easily
via homebrew


```
brew install boost
brew install pyenv
brew install pyenv-virtualenv
```

Then get the ```esig``` source code as a ```.tar.gz``` file.  You can download
the latest released version from the [PyPi downloads page](https://pypi.org/project/esig/#files).

### Python versions

The ```esig``` package is currently built for the python versions listed in
```python_versions.txt```.  This file can be modified as necessary, with
one python version number per line.
To install all these python versions via pyenv, you can use the script
```
source install_all_python_versions.sh
```

### Building

Assuming you have run the script above to install all the python versions
listed in ```python_versions.py```, you can just run the script
```
source mac_wheel_builder.sh
```
and it will loop through all the python versions, and for each one:
 * create and activate a
virtual environment
 * install python dependencies
 * compile the python "wheel"
 * "delocate" the wheel, i.e. combine other dependencies, to give a portable binary.
The output wheels will be in the ```output/``` subdirectory.
