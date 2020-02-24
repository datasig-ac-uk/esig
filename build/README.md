Helper scripts and dockerfiles to help build the Python wheels for different architectures and Python versions.

There are subdirectories for Linux, OSX, and Windows. See the README.md in each of those for instructions on building for that platform. (Note that these subdirectories correspond to the target OS for the binaries as opposed to the machine you are running.  It is possible, thanks to Docker, to build the Linux binaries from Linux, OSX, or Windows. However, at present the OSX and Windows binaries can only be built from their respective platforms.)

CI is setup for Linux and OSX builds via GitHub actions; see the `.yml` files in `/.github/workflows`. CI for Windows is still to do. Current build status is below.

### Linux

32-bit and 64-bit builds are fine.

### Windows

Windows currently fails to build in the Azure VM. Current status:

- prebuilt Docker images fail with ``filesystem layer verification failed for digest`` error (possibly to do with Azure storage)
- building the Docker images from the new `mcr.microsoft.com/dotnet/framework/sdk:4.8` base image now works
- running the build in the built image fails with two (hopefully minor errors)
  - no module named `pyparsing` building wheel
  - can’t find Visual Studio C++ 14.0

### OSX

OSX builds are fine except for Python 2.7.10 and 3.4.8, which fail with:

````
  src/Cpp_ToSig.cpp:5:10: fatal error: 'string' file not found
  #include <string>
           ^~~~~~~~
  2 warnings and 1 error generated.
  error: command 'gcc' failed with exit status 1
````
