from enum import Enum
import os
from os.path import expanduser
import sys
import platform
import textwrap

from distutils import sysconfig
from setuptools.dist import Distribution


#
# Auxiliary classes used by setup.py.
#

# Tells setuptools that package is a binary distribution
# https://lucumr.pocoo.org/2014/1/27/python-on-wheels/
class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


class InstallationConfiguration(object):
    """
    The installation configuration class, providing details required for installing esig.
    """
    def __init__(self, package_abs_root):
        print("Starting esig installer...")
        self.__package_abs_root = package_abs_root

        reported_platform = platform.system().lower()
        platform_map = {
            'windows': PLATFORM.WINDOWS,
            'linux': PLATFORM.LINUX,
            'linux2': PLATFORM.LINUX,
            'darwin': PLATFORM.MACOS,
        }

        if reported_platform not in platform_map.keys():
            raise Exception(reported_platform + " not a recognised platform.")

        self.platform = platform_map[reported_platform]
        self.is64bit = sys.maxsize > 2**32

    @property
    def include_dirs(self):
        """
        Return list of directories to search for include files.

        Returns:
            list of strings.
        """
        # libalgebra and recombine sources + wrapper code for Python.
        # TODO: remove dependency on "recombine" having been cloned into /build/recombine

        dirs = [
            os.path.join(".", "src"),
            os.path.join(".", "libalgebra"),
            os.path.join(".", "recombine"),
            os.path.join(".", "build", "recombine", "recombine")
        ]

        if 'BOOST_ROOT' in os.environ:
            dirs.append(os.environ['BOOST_ROOT'])

        if self.platform == PLATFORM.MACOS:
            dirs.append('/opt/local/include/')
        elif self.platform == PLATFORM.LINUX:
            dirs.append('/usr/include/')

        return dirs

    @property
    def library_dirs(self):
        """
        Return a list of directories to search for libraries.

        Returns:
            list of strings.
        """
        if not 'BOOST_ROOT' in os.environ:
            boost_root_env = None
        else:
            boost_root_env = os.environ['BOOST_ROOT']

        dirs = []

        if self.platform == PLATFORM.WINDOWS:
            if self.is64bit:
                lib1, lib2 = 'lib64', 'x64'
            else:
                lib1, lib2 = 'lib32', 'win32'

            if not('MKLROOT' in os.environ):
                raise RuntimeError("MKLROOT not defined.")

            # not sure why these are only needed on Windows
            if boost_root_env is not None:
                dirs.append(os.path.join(boost_root_env, lib1 + '-msvc-14.0'))
                dirs.append(os.path.join(boost_root_env, lib2, 'lib'))

            dirs.append(os.path.join(os.environ['MKLROOT'], "lib", "intel64"))
            # todo: lose hardcoded knowledge of recombine installation dir
            dirs.append(os.path.join(expanduser("~"), "lyonstech", "lib"))

        elif self.platform == PLATFORM.MACOS:
            if 'DYLD_LIBRARY_PATH' in os.environ:
                dirs.append(os.environ['DYLD_LIBRARY_PATH'].split(os.pathsep))

        elif self.platform == PLATFORM.LINUX:
            if 'LD_LIBRARY_PATH' in os.environ:
                dirs.append(os.environ['LD_LIBRARY_PATH'].split(os.pathsep))

        return dirs

	# Python extension code built with distutils is compiled with the same set of compiler options,
	# regardless of whether it's C or C++. We use C _and_ C++, which rules out certain compiler options.
    @property
    def extra_compile_args(self):
        """
        Returns a list of additional platform/compiler-dependent compiler arguments.

        Returns:
            list of strings.
        """
        args = []

        if self.platform == PLATFORM.WINDOWS:
            args.append('/EHsc')
            args.append('/DWINVER=0x0601')
            args.append('/D_WIN32_WINNT=0x0601')
            args.append('/D_SCL_SECURE_NO_WARNINGS')
            args.append('/bigobj')
        else:
            # Clang will reject this when compiling C
            if self.platform == PLATFORM.LINUX:
                args.append('-std=c++11') # want c99 as well, but not possible (see above)
            args.append('-Wno-unused-but-set-variable') # moans on some platforms

        return args

    @property
    def linker_args(self):
        """
        Returns a list of additional platform/compiler-dependent linker arguments.
        """
        args = []

        # How can we statically link for MACOS/LINUX? -static does not work on Linux.

        if self.platform == PLATFORM.MACOS:
            args.append('-static')

        return args

    @property
    def esig_version(self):
        """
        Extract the version number from the VERSION file found in the package root.

        Returns:
            str: a string representing the version number, in the format MAJOR.MINOR.RELEASE.
        """
        version_path = os.path.join(self.__package_abs_root, 'esig', 'VERSION')

        with open(version_path, 'r') as version_file:
            return (version_file.read().strip()).replace(' ', '.')

    @property
    def long_description(self):
        """
        Extract the contents of the README.md file found in the package root.

        Returns:
            str: a string representing the readme file.
        """
        readme_path = os.path.join(self.__package_abs_root, 'README.md')

        with open(readme_path, 'r') as f:
            return f.read()

    @property
    def used_libraries(self):
        """
        Returns a list of libraries that are used by esig.
        Note that on Windows, library selection is done automatically, therefore no libraries are required.

        Returns:
            list: list of strings, with each string representing a library used
        """
        libs = {
            PLATFORM.WINDOWS: ['recombine'],
            # on the Mac, recombine is a framework, not a library; needs special treatment in setup.py
            PLATFORM.MACOS: ['boost_system-mt','boost_thread-mt'],
            PLATFORM.LINUX: ['boost_system','boost_thread', 'recombine'],
        }

        return libs[self.platform]


class PLATFORM(Enum):
    WINDOWS = 1
    LINUX = 2
    MACOS = 3
