Helper scripts and dockerfiles to help build the Python wheels for different architectures and Python versions.

There are subdirectories for Linux, OSX, and Windows. See the README.md in each of those for instructions on building for that platform. (Note that these subdirectories correspond to the target OS for the binaries as opposed to the machine you are running.  It is possible, thanks to Docker, to build the Linux binaries from Linux, OSX, or Windows. However, at present the OSX and Windows binaries can only be built from their respective platforms.)

Because of the complexity of the build process there is no CI set up yet for this project. Current status of manual builds:

| | Linux x86_64        | Linux i686           | OSX  | Windows |
| :-------------: | :-------------: |:-------------:| :-----:| :-----:|
| Python 2.7 | ❌ | ✔️  |   | |
| Python 3.4 | ❌ | ✔️  |   |  |
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

Python 3.4-3.7 error on Linux x86_64:
````
  src/libalgebra/_tensor_basis.h: In instantiation of ‘alg::_tensor_basis<No_Letters, DEPTH> alg::_tensor_basis<No_Letters, DEPTH>::operator*(const alg::_tensor_basis<No_Letters, DEPTH>&) const [with unsigned int No_Letters = 20u; unsigned int DEPTH = 3u]’:
  src/libalgebra/tensor_basis.h:150:23:   required from ‘alg::tensor_basis<SCA, n_letters, max_degree>::KEY alg::tensor_basis<SCA, n_letters, max_degree>::nextkey(const KEY&) const [with SCA = double; unsigned int n_letters = 20u; unsigned int max_degree = 3u; alg::tensor_basis<SCA, n_letters, max_degree>::KEY = alg::_tensor_basis<20u, 3u>]’
  src/ToSig.cpp:219:31:   required from ‘std::string {anonymous}::tensorbasis2stringT() [with long unsigned int WIDTH = 20ul; long unsigned int DEPTH = 3ul; std::string = std::basic_string<char>]’
  src/switch.h:500:16:   required from here
  src/libalgebra/_tensor_basis.h:202:70: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
     reinterpret_cast<fp_info<word_t>::unsigned_int_type&>(dPowerOfTwo) &= fp_info<word_t>::mantissa_mask_zeroes;
                                                                        ^
  gcc: internal compiler error: Killed (program cc1plus)
````
This is the failing `gcc` command line:
````gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/python/cp35-cp35m/include/python3.5m -I./src/ -I/usr/include/boost148 -I/usr/include/ -I/opt/python/cp35-cp35m/include/python3.5m -I/opt/python/cp35-cp35m/lib/python3.5/site-packages/numpy/core/include -c src/ToSig.cpp -o build/temp.linux-x86_64-3.5/src/ToSig.o -Wno-unused-but-set-variable
````
