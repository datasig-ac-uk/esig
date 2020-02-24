## Building esig for Windows

### Prerequisites

- ensure [Docker for Windows](https://www.docker.com/docker-windows) is installed
- ensure `libalgebra` submodule has been cloned

### Building on Windows

Run the literate script `build-wheels.ps1`.

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
