from enum import Enum
import os
from os.path import expanduser
import platform
from setuptools.dist import Distribution
import sys
import textwrap


# Auxiliary classes used by setup.py.


# Tell setuptools that package is a binary distribution.
# https://lucumr.pocoo.org/2014/1/27/python-on-wheels/
class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


class InstallationConfiguration(object):
    """
    Installation configuration class, providing details required to install esig.
    """
    def __init__(self, package_root):
        print("Starting esig installer...")
        self.package_root = package_root

        reported_platform = platform.system().lower()
        platform_map = {
            'windows': PLATFORM.WINDOWS,
            'linux': PLATFORM.LINUX,
            'linux2': PLATFORM.LINUX,
            'darwin': PLATFORM.MACOS,
        }

        if reported_platform not in platform_map.keys():
            raise Exception(reported_platform + " not a recognised platform.")

        self.no_recombine = "ESIG_WITH_RECOMBINE" not in os.environ

        self.platform = platform_map[reported_platform]
        self.is64bit = sys.maxsize > 2**32

    @property
    def include_dirs(self):
        """
        Return list of directories to search for include files.

        Returns:
            list of directory paths.
        """
        # libalgebra and recombine sources + wrapper code for Python.
        # TODO: remove dependency on "recombine" having been cloned into /build/recombine

        dirs = [
            os.path.join(".", "src"),
            os.path.join(".", "libalgebra"),
        ]

        if not self.no_recombine:
            dirs.extend([
                os.path.join(".", "recombine"),
                os.path.join(".", "build", "recombine", "recombine")
            ])

        if 'BOOST_ROOT' in os.environ and os.environ['BOOST_ROOT'] != '':
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
            list of directory paths.
        """
        if not 'BOOST_ROOT' in os.environ:
            boost_rootenv = ""
        else:
            boost_root_env = os.environ['BOOST_ROOT']

        dirs = []

        if self.platform == PLATFORM.WINDOWS:
            if self.is64bit:
                lib1, lib2 = 'lib64', 'x64'
            else:
                lib1, lib2 = 'lib32', 'win32'

            dirs.append(os.path.join(boost_root_env, lib1 + '-msvc-14.0'))
            dirs.append(os.path.join(boost_root_env, lib2, 'lib'))

            # ouch, depend on MKL having been installed by rcombine
            dirs.append(
               os.path.join(
                  os.environ['USERPROFILE'],
                  "Miniconda3",
                  "pkgs",
                  "mkl-static-2021.1.1-intel_52",
                  "Library",
                  "lib"
               )
            )

            # todo: lose hardcoded knowledge of recombine installation dir
            dirs.append(os.path.join(expanduser("~"), "lyonstech", "lib"))

        # On a Mac, our best guess for including libraries will be from /opt/local/lib.
        # This is where Macports and Homebrew installs libraries to.
        elif self.platform == PLATFORM.MACOS:
            if 'DYLD_LIBRARY_PATH' in os.environ and os.environ['DYLD_LIBRARY_PATH'] != '':
                dirs = dirs + os.environ['DYLD_LIBRARY_PATH'].split(os.pathsep)

        elif self.platform == PLATFORM.LINUX:
            if 'LD_LIBRARY_PATH' in os.environ and os.environ['LD_LIBRARY_PATH'] != '':
                dirs = dirs + os.environ['LD_LIBRARY_PATH'].split(os.pathsep)

        return dirs

    @property
    def define_macros(self):
        args = []
        if self.no_recombine:
            args.append(("ESIG_NO_RECOMBINE", None))
        return args

	# Python extension code built with distutils is compiled with the same set of compiler options,
	# regardless of whether it's C or C++. We use C _and_ C++, which rules out certain compiler options.
    @property
    def extra_compile_args(self):
        """
        Returns list of additional platform/compiler-dependent compiler arguments.

        Returns:
            list of string arguments.
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
        Return list of additional platform/compiler-dependent linker arguments.
        """
        args = []

        # How can we statically link for MACOS/LINUX? -static does not work on Linux.
        if self.platform == PLATFORM.MACOS:
            args.append('-static')

        return args

    @property
    def esig_version(self):
        """
        Extract version number from VERSION file found in package root.

        Returns:
            string representing version number, in format MAJOR.MINOR.PATCH.
        """
        version_path = os.path.join(self.package_root, 'esig', 'VERSION')

        with open(version_path, 'r') as version_file:
            return (version_file.read().strip()).replace(' ', '.')

    @property
    def long_description(self):
        """
        Extract contents of README.md file found in package root.

        Returns:
            string contents of file.
        """
        readme_path = os.path.join(self.package_root, 'README.md')

        with open(readme_path, 'r') as f:
            return f.read()

    @property
    def used_libraries(self):
        """
        List libraries used by esig. On Windows, library selection is automatic, so Boost isn't included.

        Returns:
            list of library names
        """
        if self.platform == PLATFORM.WINDOWS:
            libs = []
            if not self.no_recombine:
                libs.append("recombine")
            return libs
        elif self.platform == PLATFORM.MACOS:
            return ["boost_system-mt", "boost_thread-mt"]
        elif self.platform == PLATFORM.LINUX:
            libs = ["boost_system", "boost_thread"]
            if not self.no_recombine:
                libs.append("recombine")
            return libs

        raise RuntimeError("Platform has no libraries defined")


class PLATFORM(Enum):
    WINDOWS = 1
    LINUX = 2
    MACOS = 3
