from __future__ import print_function  # Include support for Python 2.7.x.
import os
import sys
import fnmatch
import platform
import textwrap

from glob import glob as _glob

from distutils import sysconfig
from subprocess import Popen, PIPE
from setuptools.dist import Distribution
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext


#
# Auxiliary classes and helper functions to assist with the building of esig.
# These are used by the setup.py module -- which is in turn processed by setuptools.
#

# Helper function for globbing path components.
def glob(*parts):
    return _glob(os.path.join(*parts))


def message_printer(message, is_failure=False, terminate=False):
    """
    Prints a message to either stdout or stderr (depending upon the argument configuration).
    Message can be a single line (string) or multi-line (list of strings).
    If the message is indicating a failure, set is_failure to True to report to stderr. If is_failure is set to True, a general
    message will also be displayed to the user indicating commonly occurring issues that may be preventing successful execution.
    If the displayed message indicates that the installer cannot continue, setting terminate to True will stop the process.
    This will occur after printing the message -- and the installer will stop with an error code of 1.

    Args:
        message (string): a single line of text to display.
        message (list): a list of strings, each string representing a line of the output message.
        is_failure (boolean): indicates whether the message represents a failure (and thus determines whether to use stdout or stderr).
        terminate (boolean): denotes whether the installer should stop after printing the message.
    """
    MESSAGE_PREFIX = 'esig_install> '  # Prefix appended to every message displayed by this module.

    error_message = textwrap.dedent(
    """If you see this message, compilation and installation unfortunately failed. If
       building from source, the esig package requires the Boost C++ library to be
       installed and correctly configured. Installation most likely failed because the
       installer was not able to locate Boost.

       Please make sure you have done the following.
           * Downloaded and installed a copy of Boost.
           How you do this varies from platform to platorm.
           * Provided the necessary paths to Boost include files and libraries.
           This is only necessary when installing Boost to a non-standard
           location.
           If you do this, you can use the --include-paths and --library-paths
           arguments.

       Once you have done this, esig should install successfully. For more
       information, you can refer to the online documentation. It's available at
       http://esig.readthedocs.io/en/latest/troubleshooting.html.""")

    display_message = ""

    if is_failure:
        display_message = "{prefix}Something went wrong! Please review the messages below to rectify the problem.{linebreak}{prefix}{linebreak}".format(
            prefix=MESSAGE_PREFIX,
            linebreak=os.linesep,)

    if type(message) == list:
        for line in message:
            display_message = ("{display_message}{prefix}{line}{linebreak}").format(
                                    display_message=display_message,
                                    linebreak=os.linesep,
                                    prefix=MESSAGE_PREFIX,
                                    line=line)

        display_message = display_message.rstrip(os.linesep)
    else:
        display_message = "{display_message}{prefix}{message}".format(
                            prefix=MESSAGE_PREFIX,
                            display_message=display_message,
                            linebreak=os.linesep,
                            message=message,)

    if is_failure or terminate:
        display_message = '{display_message}{linebreak}'.format(
            display_message=display_message,
            linebreak=os.linesep,)

    if is_failure:
        display_message = ("{display_message}{prefix}{linebreak}"
                           "{prefix}General Information{linebreak}"
                           "{prefix}==================={linebreak}"
                           "{error_message}").format(
                               display_message=display_message,
                               prefix=MESSAGE_PREFIX,
                               linebreak=os.linesep,
                               error_message=error_message)

        print_destination = sys.stderr
    else:
        print_destination = sys.stdout

    if terminate:
        display_message = "{display_message}{prefix}{linebreak}{prefix}Now terminating.{linebreak}".format(
                                display_message=display_message,
                                linebreak=os.linesep,
                                prefix=MESSAGE_PREFIX,)

    print(display_message, file=print_destination)

    if terminate:
        sys.exit(1)


def get_platform():
    """
    Returns an enum specifying the type of operating system currently used.

    Args:
        None
    Returns:
        PLATFORMS: an enum representation of the platform currently used (WINDOWS, LINUX, MACOS).
    """
    reported_platform = platform.system().lower()

    platform_enums = {
        'windows': PLATFORMS.WINDOWS,
        'linux': PLATFORMS.LINUX,
        'linux2': PLATFORMS.LINUX,
        'darwin': PLATFORMS.MACOS,
    }

    if reported_platform not in platform_enums.keys():
        return PLATFORMS.OTHER

    return platform_enums[reported_platform]


# Tells setuptools that package is a binary distribution
# https://lucumr.pocoo.org/2014/1/27/python-on-wheels/
class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


class InstallExtensionCommand(install):
    """
    Runs a series of checks to determine if the package can be successfully installed and run.
    Extends the install class.
    """
    user_options = install.user_options + [
        ('include-dirs=', None, None),
        ('library-dirs=', None, None)
    ]


    def initialize_options(self):
        install.initialize_options(self)
        self.include_dirs = ''
        self.library_dirs = ''


    def finalize_options(self):
        if self.include_dirs != '':
            CONFIGURATION.include_dirs = self.include_dirs

        if self.library_dirs != '':
            CONFIGURATION.library_dirs = self.library_dirs

        install.finalize_options(self)


class Enum(set):
    """
    A simple class emulating an enum type.
    Extends the set type.
    """
    def __getattr__(self, name):
        """
        Given the name specified, returns the enum value corresponding to that name.
        If the name does not exist, an AttributeError is raised.

        Args:
            self (Enum): an instance of type Enum
            name (string): the name of the enum to find
        Returns:
            string: the string representation (if it exists)
        Raises:
            AttributeError: raised if the enum name cannot be found within the instance.
        """
        if name in self:
            return name

        raise AttributeError


class InstallationConfiguration(object):
    """
    The installation configuration class, an instance of which provides the relevant details required for installing the esig package.
    """
    def __init__(self, package_abs_root):
        message_printer("Starting esig installer...", True)
        self.__package_abs_root = package_abs_root
        self.__include_dirs = None
        self.__library_dirs = None


    @property
    def platform(self):
        """
        Returns an enum specifying the type of operating system currently used.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
        Returns:
            PLATFORMS: an enum representation of the platform currently used (WINDOWS, LINUX, MACOS).
        """
        return get_platform()


    @property
    def is_x64(self):
        """
        Returns a boolean indicating whether the current platform is 64-bit (True) or not (False).

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
        Returns:
            bool: True if the platform is x64, False otherwise.
        """
        return sys.maxsize > 2**32

    @property
    def include_dirs(self):
        """
        Returns a list of directories with which to search for include files.
        A series of hard-coded included paths are appended to the end of the stored list.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
        Returns:
            list: a list of strings, with each string denoting a path.
        """
        return_list = []

        # Add Python's include directory.
        return_list.append(sysconfig.get_python_inc())

        # libalgebra and recombine sources + wrapper code for Python.
        # TODO: use standard os.path.join method here
        # TODO: remove dependency on "recombine" having been cloned into /build/recombine
        #if self.platform == PLATFORMS.WINDOWS:
        #    return_list.append('.\\src\\')
        #    return_list.append('.\\libalgebra\\')
        #    return_list.append('.\\recombine\\')
        #    return_list.append('.\\build\\recombine\\recombine\\')
        #else:
        #    return_list.append('./src/')
        #    return_list.append('./libalgebra/')
        #    return_list.append('./recombine/')
        #    return_list.append('./build/recombine/recombine/')
        _jn = os.path.join

        # Platform independent joining. We probably don't need to use a join at all.
        return_list.extend([
            _jn(".", "src"),
            _jn(".", "libalgebra"),
            _jn(".", "recombine"),
            _jn(".", "build", "recombine", "recombine")
        ])
        del _jn

        # Append any command-line supplied arguments to the list
        if self.__include_dirs is not None:
            return_list = return_list + self.__include_dirs

        # Add the contents of the CPATH environment variable.
        if 'CPATH' in os.environ and os.environ['CPATH'] != '':
            return_list = os.environ['CPATH'].split(os.pathsep)

        # This is now based upon Terry's code, to include standard locations for Boost.
        if 'BOOST_ROOT' in os.environ and os.environ['BOOST_ROOT'] != '':
            return_list.append(os.environ['BOOST_ROOT'])

        # On a Mac, Macports/Homebrew is likely to install headers to this location.
        if self.platform == PLATFORMS.MACOS:
            return_list.append('/opt/local/include/')
        # Fallback for not MacOS or Windows. Assume Linux.
        elif self.platform != PLATFORMS.WINDOWS:
            return_list.append('/usr/include/')

        return return_list


    @include_dirs.setter
    def include_dirs(self, paths):
        """
        Sets the internal attribute for directories to include. Accepts a list of strings.
        This is used if the additional command-line argument --include-dirs is provided.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
            paths (list): a list of strings, with each string representing a path
        Returns:
            None
        """
        if paths != '':
            self.__include_dirs = paths.split(os.pathsep)


    @property
    def library_dirs(self):
        """
        Returns a list of directories with which to search for libraries.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
        Returns:
            list: a list of strings, with each string denoting a path to a library include directory.
        """
        return_list = []

        # TODO: why mixture of + and append? Do they both mean 'append'?
        # Append any command-line supplied paths to the list
        if self.__library_dirs is not None:
            return_list = return_list + self.__library_dirs

        # Jump into Terry's old code.
        boost_root_env = ''
        python_version = sys.version_info

        # Obtain the BOOST_ROOT environment variable.
        if 'BOOST_ROOT' in os.environ and os.environ['BOOST_ROOT'] != '':
            boost_root_env = os.environ['BOOST_ROOT']

        # On Windows, we can follow a pretty set-in-stone path structure, all hailing from BOOST_ROOT.
        # The version of the MSVC compiler depends upon the version of Python being used.
        if self.platform == PLATFORMS.WINDOWS:
            lib_directory = {
                True: ('lib64', 'x64'),
                False: ('lib32', 'win32'),
            }

            if python_version[0] == 3 and python_version[1] < 5:
                compiler_version = 'msvc-10.0'
            else:
                compiler_version = 'msvc-14.0'

            return_list.append(os.path.join(boost_root_env, '{lib_directory}-{compiler_version}'.format(lib_directory=lib_directory[self.is_x64][0], compiler_version=compiler_version)))
            return_list.append(os.path.join(boost_root_env, lib_directory[self.is_x64][1], 'lib'))
            if not('MKLROOT' in os.environ):
                raise RuntimeError("MKLROOT not defined.") #NOTE: errors must derive from Exception. I've wrapped this in a RuntimeError
            # not sure why this is only needed on Windows
            return_list.append(os.path.join(os.environ['MKLROOT'], "lib", "intel64"))
            # todo: lose hardcoded knowledge of recombine installation dir
            from os.path import expanduser
            recombine_lib_dir = os.path.join(expanduser("~"), "lyonstech", "lib")
            #os.listdir(recombine_lib_dir)
            return_list.append(recombine_lib_dir)

        # On a Mac, our best guess for including libraries will be from /opt/local/lib.
        # This is where Macports and Homebrew installs libraries to.
        elif self.platform == PLATFORMS.MACOS:
            return_list.append('/opt/local/lib/')

            if 'DYLD_LIBRARY_PATH' in os.environ and os.environ['DYLD_LIBRARY_PATH'] != '':
                return_list = return_list + os.environ['DYLD_LIBRARY_PATH'].split(os.pathsep)
        elif self.platform == PLATFORMS.LINUX:
            include_directory = {
                True: 'x86_64',
                False: 'i386',
            }

            return_list.append('/lib/{architecture}-linux-gnu/'.format(architecture=include_directory[self.is_x64]))
            return_list.append('/usr/lib/{architecture}-linux-gnu/'.format(architecture=include_directory[self.is_x64]))

            if 'LD_LIBRARY_PATH' in os.environ and os.environ['LD_LIBRARY_PATH'] != '':
                return_list = return_list + os.environ['LD_LIBRARY_PATH'].split(os.pathsep)
        return return_list


    @library_dirs.setter
    def library_dirs(self, paths):
        """
        Sets the internal attribute for directories in which libraries may be found.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
            paths (list): a list of strings, with each string representing a path
        Returns:
            None
        """
        if paths != '':
            self.__library_dirs = paths.split(os.pathsep)


	# Python extension code built with distutils is compiled with the same set of compiler options,
	# regardless of whether it's C or C++. We use C _and_ C++, which rules out certain compiler options.
    @property
    def extra_compile_args(self):
        """
        Returns a list of additional arguments that must be supplied to the compiler.
        Platform and compiler dependent.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
        Returns:
            list: a list of strings, each string representing an argument to supply to the compiler.
        """
        args = []

        if self.platform == PLATFORMS.WINDOWS:
            args.append('/EHsc')
            args.append('/DWINVER=0x0601')
            args.append('/D_WIN32_WINNT=0x0601')
            args.append('/D_SCL_SECURE_NO_WARNINGS')
            args.append('/bigobj')
        else:
            # Clang will reject this when compiling C
            if self.platform == PLATFORMS.LINUX:
                args.append('-std=c++11') # want c99 as well, but not possible (see above)
            args.append('-Wno-unused-but-set-variable') # moans on some platforms

        return args


    @property
    def linker_args(self):
        """
        Returns a list of additional arguments that must be used by the linker.
        Returned lists are platform and compiler dependent.
        """
        args = []

        # How can we statically link for MACOS/LINUX? -static does not work on Linux.

        if self.platform == PLATFORMS.MACOS:
            args.append('-static')

        return args


    @property
    def esig_version(self):
        """
        Returns a string representing the version number of esig.
        The version is extracted from the VERSION file, found in the base of the package.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
        Returns:
            str: a string representing the version number, in the format MAJOR.MINOR.RELEASE.
        """
        version_path = os.path.join(self.__package_abs_root, 'esig', 'VERSION')

        with open(version_path, 'r') as version_file:
            version = (version_file.read().strip()).replace(' ', '.')

        return version


    @property
    def long_description(self):
        """
        Returns a string representing the long description for the esig package.
        The long description of the contents of the README.md file in the base of the package.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
        Returns:
            str: a string representing the readme file.
        """
        readme_path = os.path.join(self.__package_abs_root, 'README.md')

        with open(readme_path, 'r') as f:
            long_description = f.read()

        return long_description


    @property
    def used_libraries(self):
        """
        Returns a list of libraries that are used by esig.
        Note that on Windows, library selection is done automatically, therefore no libraries are required.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
        Returns:
            list: list of strings, with each string representing a library used
        """
        libs = {
            PLATFORMS.WINDOWS: ['recombine'],
            # on the Mac, recombine is a framework, not a library; needs special treatment in setup.py
            PLATFORMS.MACOS: ['boost_system-mt','boost_thread-mt'],
            PLATFORMS.LINUX: ['boost_system','boost_thread', 'recombine'],
            PLATFORMS.OTHER: ['boost_system','boost_thread'],
        }

        return libs[self.platform]


PLATFORMS = Enum(['WINDOWS', 'LINUX', 'MACOS', 'OTHER'])
