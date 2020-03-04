Helper scripts and dockerfiles to help build the Python wheels for different architectures and Python versions.

There are subdirectories for Linux, OSX, and Windows. See the README.md in each of those for instructions on building for that platform. (Note that these subdirectories correspond to the target OS for the binaries as opposed to the machine you are running.  It is possible, thanks to Docker, to build the Linux binaries from Linux, OSX, or Windows. However, at present the OSX and Windows binaries can only be built from their respective platforms.)

Getting a reproducible, fully automated build has proved difficult despite Nick's excellent starting point. Current status and outstanding problems are summarised below:

### Linux :white_check_mark:

32-bit and 64-bit builds are fine and now fully automated. They run in the `esig` repo as follows:
1. build Docker file for each architecture to obtain suitable container
1. create Python `sdist` archive from sources in the repo
1. for each architecture, instantiate container and then:
   - for each Python version, build wheel from the `sdist` archive
   
This process runs as a [GitHub Action](https://github.com/alan-turing-institute/esig/actions?query=workflow%3Abuild-OSX) specified in `/.github/workflows/build-Linux.yml`.

Issues addressed:
| Task | Completed |
| ---- | --------- |
| [Automate OSX and Linux builds through GitHub actions](https://github.com/alan-turing-institute/esig/issues/18) | 25 Feb 2020 |
| [64-bit Linux build working with Python 2.7-3.8]() | 24 Feb 2020 |
| [32-bit Linux build working with Python 2.7-3.8](https://github.com/alan-turing-institute/esig/issues/14) | 18 Feb 2020 |
| [Replace libalgebra files by submodule](https://github.com/alan-turing-institute/esig/issues/6) | 18 Feb 2020 |
| [Import esig source code from pypi.org](https://github.com/alan-turing-institute/esig/issues/5) | 3 Feb 2020 |

### Windows :x:

Windows currently fails to build in the Azure VM. Current status:

- Nick's prebuilt Docker images fail with ``filesystem layer verification failed for digest`` error (possibly to do with Azure storage)
- building the Docker images from the new `mcr.microsoft.com/dotnet/framework/sdk:4.8` base image now works
- development process is quite slow - for example
- running the build in the built image fails with two (hopefully minor errors)
  - no module named `pyparsing` building wheel
  - can’t find Visual Studio C++ 14.0

### OSX :white_check_mark: :x:

OSX builds are fine except for Python 2.7.10 and 3.4.8, which fail with:

````
  src/Cpp_ToSig.cpp:5:10: fatal error: 'string' file not found
  #include <string>
           ^~~~~~~~
  2 warnings and 1 error generated.
  error: command 'gcc' failed with exit status 1
````
Issues addressed:
| Task | Completed |
| ---- | --------- |
| [OSX build working for Python 3.5-3.8](https://github.com/alan-turing-institute/esig/issues/16) | 18 Feb 2020 |
