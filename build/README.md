Helper scripts and dockerfiles to help build the Python wheels for different architectures and Python versions.

There are subdirectories for Linux, OSX, and Windows. See the README.md in each of those for instructions on building for that platform. (Note that these subdirectories correspond to the target OS for the binaries as opposed to the machine you are running.  It is possible, thanks to Docker, to build the Linux binaries from Linux, OSX, or Windows. However, at present the OSX and Windows binaries can only be built from their respective platforms.)

Because of the complexity of the build process there is no CI set up yet for this project. Current status of manual builds:

| | Linux x86_64        | Linux i686           | OSX  | Windows |
| :-------------: | :-------------: |:-------------:| :-----:| :-----:|
| Python 2.7 | ✔️ | ✔️  |   | |
| Python 3.4 | ✔️ | ✔️  |   |  |
| Python 3.5 | ✔️ | ✔️  | ✔️  | ❌ | 
| Python 3.6 | ✔️ | ✔️  | ✔️  | ❌ | 
| Python 3.7 | ✔️ | ✔️  | ✔️  | ❌ | 
| Python 3.8 | ✔️ | ✔️  | ✔️  | ❌ | 
