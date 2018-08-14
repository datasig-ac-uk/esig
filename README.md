## esig_builder

[esig](http://esig.readthedocs.io/en/latest/index.html) is a python package
written by Terry Lyons et al., for transforming vector time series in
stream space to signatures in effect space.

This package contains helper scripts and dockerfiles to help build the
package (i.e. create pre-compiled python ***wheels***) for different
architectures and python versions.

There are subdirectories in this repo for Linux, OSX, and Windows.  Please see
the README.md in each of those for instructions on building for that
OS.
(Note that these subdirectories correspond to the target OS for the binaries as opposed
to the machine you are running.  It is
possible (thanks to Docker) to build the Linux binaries from Linux, OSX, or Windows.
However, at present the OSX binaries can only be built from OSX, and the Windows binaries can only
be built from Windows.)
