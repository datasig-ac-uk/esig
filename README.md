# esig
The Python package [esig](https://pypi.org/project/esig/) provides a toolset (previously called sigtools) for transforming vector time series in stream space to signatures in effect space. It is based on the [libalgebra](https://github.com/terrylyons/libalgebra) C++ library.

[![build](https://github.com/datasig-ac-uk/esig/actions/workflows/build.yml/badge.svg?branch=release)](https://github.com/datasig-ac-uk/esig/actions/workflows/build.yml)

## Installation
esig can be installed from a wheel using pip in most cases.
The wheels contain all of the dependencies and thus make it easy to use the package.
For example, on Python 3.8, you can install esig using the following console command:
```
python3.8 -m pip install esig
```
(You may need to tweak this command based on your platform, Python version, and preferences.)

esig can be compiled from source, but this is not advised.
More information can be found in the [documentation](https://esig.readthedocs.org/en/latest).

## Basic usage
esig provides a collection of basic functions for computing the signature of a data stream in the form of a Numpy array.
The `stream2sig` function computes the signature of a data stream up to a specific depth.
For example, we can create a very simple data stream and compute its signature as follows.
```python3
import numpy as np
import esig

stream = np.array([
    [1.0, 1.0],
    [3.0, 4.0],
    [5.0, 2.0],
    [8.0, 6.0]
])
depth = 2

sig = esig.stream2sig(stream, depth) # compute the signature
print(sig) # prints "[1.0, 7.0, 5.0, 24.5, 19.0, 16.0, 12.5]"
```
The signature is returned as a flat Numpy array that contains the terms of the signature - which is fundamentally a higher dimensional tensor - in degree order.
This first element is always 1.0, which corresponds to the empty tensor key.
In this case the dimension is 2 (specified by the number of columns in the stream array), and so the next two elements are the signature elements corresponding to the words (1) and (2).
These are the depth 1 words.
The final 4 elements are the depth 2 words (1,1), (1,2), (2,1), and (2,2).
esig provides the `sigkeys` function to generate these labels for you based on the parameters of the data.
```python3
width = 2
sig_keys = esig.sigkeys(width, depth)
print(sig_keys) # prints " () (1) (2) (1,1) (1,2) (2,1) (2,2)"
```
To compute the log signature of a data stream you use the `stream2logsig` function.
This works in a similar manner to the `stream2sig` function in that it takes a Numpy array (the data) and a depth and returns a flat Numpy array containing the elements of the log signature in degree order.
```python3
log_sig = esig.stream2logsig(stream, depth)
print(log_sig) # prints "[7.  5.  1.5]"
```
Here the first two elements are the depth 1 Lie elements (corresponding to the letters 1 and 2) and the third element is the coefficient of the Hall basis element \[1,2\].
Again, esig provides a utility function `logsigkeys` for getting the keys that correspond to the coefficients in order for the log signature.
```python3
log_sig_keys = esig.logsigkeys(width, depth)
print(log_sig_keys) # prints " 1 2 [1,2]"
```
There are two additional utility functions for computing the size of a signature or logsignature with a specified dimension and depth: `sigdim` and `logsigdim`.
These functions return an integer that is the dimension of the Numpy array returned from the `stream2sig` or `stream2logsig` functions, respectively.

esig also provides another function `recombine`, which performs a reduction of a measure defined on a large ensemble in a way so that the resulting measure has the same total mass, but is supported on a (relatively) small subset of the original ensemble.
In particuar, the expected value over the ensemble with respect to the new measure agrees with that of the original measure.

### Using alternative computation backends
esig uses libalgebra as a backend for computing signatures and log signatures
 by default.
However, the computation backend can be changed to instead use an alternative
 library for computing signatures and log signatures.
This is achieved by using the `set_backend` function in esig and providing
 the name of the backed that you wish to use.
For example, we can switch to using the `iisignature` package as a backend by
 first installing the `iisignature` package and then using the command
```python3
import esig
esig.set_backend("iisignature")
```
To make it easier to install and use `iisignature` as a backend, it is
 offered as an optional extra when installing esig:
```
python3.8 -m pip install esig[iisignature]
```
You can also define your own backend for performing calculations by creating
 a class derived from `esig.backends.BackendBase`, implementing the methods
 `describe_path` (log_signature) and `signature` and related methods.
