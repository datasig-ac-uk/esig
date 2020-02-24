Helper scripts and dockerfiles to help build the Python wheels for different architectures and Python versions.

There are subdirectories for Linux, OSX, and Windows. See the README.md in each of those for instructions on building for that platform. (Note that these subdirectories correspond to the target OS for the binaries as opposed to the machine you are running.  It is possible, thanks to Docker, to build the Linux binaries from Linux, OSX, or Windows. However, at present the OSX and Windows binaries can only be built from their respective platforms.)

Because of the complexity of the build process there is no CI set up yet for this project. Current status of manual builds:

| | Linux x86_64        | Linux i686           | OSX  | Windows |
| :-------------: | :-------------: |:-------------:| :-----:| :-----:|
| Python 2.7 | ❌ |   |   | |
| Python 3.4 |  |   |   |  |
| Python 3.5 | ❌ | ✔️  | ✔️  | ❌ | 
| Python 3.6 | ❌ | ✔️  | ✔️  | ❌ | 
| Python 3.7 | ❌ | ✔️  | ✔️  | ❌ | 
| Python 3.8 | ❌ | ✔️  | ✔️  | ❌ | 

Python 2.7 error on Linux x86_64: 

````
/usr/include/boost148/boost/optional/optional.hpp:364:8: note: in expansion of macro ‘BOOST_STATIC_ASSERT’
          BOOST_STATIC_ASSERT ( ::boost::mpl::not_<is_reference_predicate>::value ) ;
          ^
  src/ToSig.cpp: In function ‘size_t {anonymous}::GetLogSigT()’:
  src/ToSig.cpp:286:24: warning: typedef ‘array’ locally defined but not used [-Wunused-local-typedefs]
     typedef const double array[WIDTH];
                          ^
  gcc: internal compiler error: Killed (program cc1plus)
````
