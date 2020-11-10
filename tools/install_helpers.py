from enum import Enum
import os
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
    The installation configuration class, an instance of which provides the relevant details required for installing the esig package.
    """
    def __init__(self, package_abs_root):
        print("Starting esig installer...")
        self.__package_abs_root = package_abs_root
        self.include_dirs = None


    @property
    def platform(self):
        """
        Returns an enum specifying the type of operating system currently used.

        Returns:
            one of PLATFORM.WINDOWS, PLATFORM.LINUX or PLATFORM.MACOS.
        """
        reported_platform = platform.system().lower()

        platform_map = {
            'windows': PLATFORM.WINDOWS,
            'linux': PLATFORM.LINUX,
            'linux2': PLATFORM.LINUX,
            'darwin': PLATFORM.MACOS,
        }

        if reported_platform not in platform_map.keys():
            raise Exception(reported_platform + " not a recognised platform.")

        return platform_enums[reported_platform]


    @property
    def is64bit(self):
        """
        Indicate whether current platform is 64-bit.

        Returns:
            bool: True if 64-bit, False otherwise.
        """
        return sys.maxsize > 2**32

    @property
    def include_dirs(self):
        """
        Return list of directories to search for include files.

        Returns:
            list of strings.
        """
        return_list = []

        # Add Python's include directory.
        return_list.append(sysconfig.get_python_inc())

        # libalgebra and recombine sources + wrapper code for Python.
        # TODO: remove dependency on "recombine" having been cloned into /build/recombine

        return_list.extend([
            os.path.join(".", "src"),
            os.path.join(".", "libalgebra"),
            os.path.join(".", "recombine"),
            os.path.join(".", "build", "recombine", "recombine")
        ])

        # Append any command-line supplied arguments to the list
        if self.include_dirs is not None:
            return_list = return_list + self.include_dirs

        # Add the contents of the CPATH environment variable.
        if 'CPATH' in os.environ and os.environ['CPATH'] != '':
            return_list = os.environ['CPATH'].split(os.pathsep)

        # This is now based upon Terry's code, to include standard locations for Boost.
        if 'BOOST_ROOT' in os.environ and os.environ['BOOST_ROOT'] != '':
            return_list.append(os.environ['BOOST_ROOT'])

        # On a Mac, Macports/Homebrew is likely to install headers to this location.
        if self.platform == PLATFORM.MACOS:
            return_list.append('/opt/local/include/')
        # Fallback for not MacOS or Windows. Assume Linux.
        elif self.platform != PLATFORM.WINDOWS:
            return_list.append('/usr/include/')

        return return_list


    @property
    def library_dirs(self):
        """
        Return a list of directories to search for libraries.

        Returns:
            list of strings.
        """
        return_list = []

        # Jump into Terry's old code.
        boost_root_env = ''
        python_version = sys.version_info

        # Obtain the BOOST_ROOT environment variable.
        if 'BOOST_ROOT' in os.environ and os.environ['BOOST_ROOT'] != '':
            boost_root_env = os.environ['BOOST_ROOT']

        # On Windows, we can follow a pretty set-in-stone path structure, all hailing from BOOST_ROOT.
        # The version of the MSVC compiler depends upon the version of Python being used.
        if self.platform == PLATFORM.WINDOWS:
            lib_directory = {
                True: ('lib64', 'x64'),
                False: ('lib32', 'win32'),
            }

            return_list.append(
                os.path.join(
                    boost_root_env,
                    '{lib_directory}-msvc-14.0'.format(lib_directory=lib_directory[self.is64bit][0])
                )
            )
            return_list.append(os.path.join(boost_root_env, lib_directory[self.is64bit][1], 'lib'))
            if not('MKLROOT' in os.environ):
                raise RuntimeError("MKLROOT not defined.")
            # not sure why this is only needed on Windows
            return_list.append(os.path.join(os.environ['MKLROOT'], "lib", "intel64"))
            # todo: lose hardcoded knowledge of recombine installation dir
            from os.path import expanduser
            recombine_lib_dir = os.path.join(expanduser("~"), "lyonstech", "lib")
            return_list.append(recombine_lib_dir)

        # On a Mac, our best guess for including libraries will be from /opt/local/lib.
        # This is where Macports and Homebrew installs libraries to.
        elif self.platform == PLATFORM.MACOS:
            return_list.append('/opt/local/lib/')

            if 'DYLD_LIBRARY_PATH' in os.environ and os.environ['DYLD_LIBRARY_PATH'] != '':
                return_list = return_list + os.environ['DYLD_LIBRARY_PATH'].split(os.pathsep)
        elif self.platform == PLATFORM.LINUX:
            pass
#            include_directory = {
#                True: 'x86_64',
#                False: 'i386',
#            }
#
#            return_list.append('/lib/{architecture}-linux-gnu/'.format(architecture=include_directory[self.is64bit]))
#            return_list.append('/usr/lib/{architecture}-linux-gnu/'.format(architecture=include_directory[self.is64bit]))
#
#            if 'LD_LIBRARY_PATH' in os.environ and os.environ['LD_LIBRARY_PATH'] != '':
#                return_list = return_list + os.environ['LD_LIBRARY_PATH'].split(os.pathsep)
        return return_list


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
