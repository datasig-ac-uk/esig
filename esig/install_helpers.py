from __future__ import print_function  # Include support for Python 2.7.x.
import os
import sys
import fnmatch
import platform

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

__author__ = 'David Maxwell <maxwelld90@gmail.com>'
__date__ = '2017-09-05'
__version__ = 5


# I like this as a little helper function for globbing path components.
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
    def get_error_message():
        """
        Loads the error message from the ERROR_MESSAGE file. Assumes that it is located in the same directory as this module.
        The message is then returned as a string.

        Args:
            None
        Returns:
            string: the error message to display.
        """
        dir_path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(dir_path, 'ERROR_MESSAGE')
        return_str = ""

        f = open(filename, 'r')

        for line in f:
            line = line.strip()

            line = "{prefix}{line}{linebreak}".format(
                prefix=MESSAGE_PREFIX,
                line=line,
                linebreak=os.linesep,
            )

            return_str = "{return_str}{line}".format(
                return_str=return_str,
                line=line,
            )

        f.close()
        return return_str


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
                               error_message=get_error_message())

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


class ComponentChecker(object):
    """
    A class providing a series of static methods to check for the existence of modules and header files,
    or whatever needs to be checked for the compilation process to work successfully.
    Fully decoupled from any other components of the installer's helpers. Noice.
    """
    _sample_includes = {
        'boost': ['boost/thread/shared_mutex.hpp'],
    }  # A series of include files that are required for the compilation process.

    _required_libraries = {
        'boost': {'system': ['libboost_system', 'libboost_system-mt'],
                  'thread': ['libboost_thread', 'libboost_thread-mt'],}
    }  # The names of libraries that are required for the compilation process. Alternative names can be used.


    @staticmethod
    def check_python_version():
        """
        Checks the version of Python. If it is less than MINIMUM_PYTHON_VERSION, the method returns False.
        Otherwise, True is returned.

        Args:
            None
        Returns:
            bool: True iif the Python version is suitable; False otherwise.
        """
        if sys.version_info < MINIMUM_PYTHON_VERSION:
            return False

        return True


    @staticmethod
    def check_libraries(paths_list):
        """
        Checks for the availability of required library files, as specified in ComponentChecker._required_libraries.
        Paths to potential locations where the libraries are stored is specified by parameter paths_list.
        If, by the end of scanning all directories, one or more required libraries cannot be found, the installer will
        display an error message explaining what to do to rectify the problem, and will terminate.

        Args:
            paths_list (list): a list of strings, with each string representing a system path to a location of libraries.
        Returns:
            None
        """
        platform = get_platform()

        checkers = {
            PLATFORMS.MACOS: ComponentChecker.__dyld_check,
            PLATFORMS.WINDOWS: ComponentChecker.__win32_check,
            PLATFORMS.LINUX: ComponentChecker.__ld_check,
            PLATFORMS.OTHER: ComponentChecker.__ld_check,
        }

        def prepare_data_structure():
            """
            Prepares a data structure for processing. Obtains data from ComponentChecker._required_libraries.
            Repurposes this initial data structure into a list of dictionaries, each dictionary containing information on
            required libraries. Includes a boolean 'found' key/value pair, storing whether finding the library was successful or not.

            Args:
                None

            Returns:
                list: list of dictionaries, representing information on the libraries to locate.
            """
            return_ds = []

            for package_name in ComponentChecker._required_libraries:
                for library_name in ComponentChecker._required_libraries[package_name]:
                    library_list = ComponentChecker._required_libraries[package_name][library_name]

                    return_ds.append({'package': package_name,
                                      'library_name': library_name,
                                      'libraries': library_list, 'found': False})

            return return_ds


        def get_missing_libraries(library_list):
            """
            Returns a list of dictionaries where the required library has not been found.
            Essentially, this function acts as a filter.

            Args:
                library_list (list): list of dictionaries representing information on libraries.

            Returns:
                list: filtered list of dictionaries, where only those where the associated library was not found are present.
            """
            return_list = []

            for library in library_list:
                if not library['found']:
                    return_list.append(library)

            return return_list

        library_dict = prepare_data_structure()

        # Iterate over the data structure, checking to see what libraries can be found.
        for library_object in library_dict:
            libraries = library_object['libraries']

            for library in libraries:
                exists = checkers[platform](paths_list, library)

                if exists:
                    library_object['found'] = True

        missing_libraries = get_missing_libraries(library_dict)

        if len(missing_libraries) > 0:
            # If this condition evaluates to True, there's a missing library and we cannot continue.
            missing_libraries_message = []

            missing_libraries_message.append("We couldn't find one or more required libraries. It (or they) are listed below.")
            missing_libraries_message.append("Please fix this problem before continuting.")
            missing_libraries_message.append("Try using the --library-dirs argument when running the installer.")
            missing_libraries_message.append("")
            missing_libraries_message.append("Missing library/libraries:")

            for library in missing_libraries:
                missing_libraries_message.append("    Package {package_name}, library {library_name} ({library_list})".format(
                    package_name=library['package'],
                    library_name=library['library_name'],
                    library_list=' or '.join(library['libraries'])
                ))

            message_printer(missing_libraries_message, is_failure=True, terminate=True)

        message_printer("Required libraries found! Hurrah!")


    @staticmethod
    def __win32_check(paths_list, search_for):
        """
        Checks for the existence of a given library file (specified by search_for) within at least one of the paths
        specified by parameter paths_list. Returns True iif a file matching the library name can be found, False otherwise.

        Args:
            paths (list): a list of strings, with each string representing an additional path to search through
            search_for (string): the library to search for.
        Returns:
            boolean: iif the library is found, True; False otherwise.
        """
        #NOTE: I don't undertand why this code. Surely it would be better to use a glob here rather than fnmatch.
        library_wildcard = '{search_for}*.lib'.format(search_for=search_for)

        for path in paths_list:
        #    try:
        #        for f in os.listdir(path):
        #            if fnmatch.fnmatch(f, library_wildcard) and f.startswith(search_for):
        #                return True
        #    except FileNotFoundError:  # This should never have occurred since f is taken from listdir.
        #        pass

            # I think this (much simpler) code does exactly the same thing.
            if glob(path, library_wildcard):
                return True

        return False

    @staticmethod
    def __ld_check(paths_list, search_for):
        """
        Checks for the existence of a given library (search_for) using ldconfig. For *NIX-based systems.
        Returns True iif the library exists within one of the paths provided in the list of paths (paths), or one of the
        default library locations.

        Args:
            paths (list): a list of strings, with each string representing an additional path to search through
            search_for (string): the library to search for.
        Returns:
            boolean: iif the library is found, True; False otherwise.
        """
        search_for = search_for.lower()
        paths_str = ''

        if paths_list is not None:
            paths_str = os.pathsep.join(paths_list)

        command = 'ldconfig -N -v {paths}'.format(paths=paths_str)
        process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)

        for line in process.stdout.readlines():
            line = line.decode('utf-8').strip()

            if search_for in line.lower():
                return True

        return False


    @staticmethod
    def __dyld_check(paths_list, search_for):
        """
        Checks for the existence of libraries using dynamic linking (.dylib), used on Darwin-based systems (macOS).
        Returns True iif the library is found; False otherwise.
        Libraries are looked for in one of the paths provided by the list paths_list.

        Args:
            paths (list): a list of strings, with each string representing an additional path to search through
            search_for (string): the name of the library to search for (e.g. libboost_system)
        Returns:
            boolean: iif the library is found, True; False otherwise.
        """
        search_for = '{filename}.dylib'.format(filename=search_for.lower())

        def run_command(path):
            """
            Runs the otool command (Darwin/macOS) to determine whether the given library exists.
            If the library is successfully found, True is returned; False otherwise.

            Args:
                path (string): path to the directory of libraries.
            Returns:
                bool: denotes whether the library has been successfully found at the given location (True) or not (False).
            """
            complete_path = os.path.join(path, search_for)
            command = 'otool -D {complete_path}'.format(complete_path=complete_path)

            process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            process.wait()

            if process.returncode == 0:
                return True

            return False

        if paths_list is None:
            return False

        for path in paths_list:
            if run_command(path):
                return True

        return False


    @staticmethod
    def check_includes(paths_list):
        """
        Checks for the availability of required include (header) files, as specified in ComponentChecker._sample_includes.
        Paths to potential locations where the headers are stored is specified by parameter paths_list.
        If, by the end of scanning all directories, one or more required headers cannot be found, the installer will
        display an error message explaining what to do to rectify the problem, and will terminate.

        Args:
            paths_list (list): a list of strings, with each string representing a system path to a location of headers.
        Returns:
            None
        """
        def prepare_data_structure():
            """
            Prepares a dictionary object for processing.
            Given the paths specified in ComponentChecker._sample_includes, returns a dictionary where keys are each
            of the files in _sample_includes. The values are all set to False, indicating that the files have not been
            located. These can be switched to True to indicate successful location of the files.

            Args:
                None

            Returns:
                dict: dictionary of sample include files as keys, with boolean values.
            """
            return_ds = {}

            for component_include_paths in ComponentChecker._sample_includes.values():
                for file_include_path in component_include_paths:
                    return_ds[file_include_path] = False

            return return_ds


        def get_missing_files(file_dict):
            """
            Returns a list of all files in the file_dict dictionary provided that have a value of False.
            This indicates that they were not found in the checks, and as such the compilation process will fail.
            If all files have been successfully found, or no files were specified in the dictionary, then an empty list is returned.

            Args:
                file_dict (dict): a dictionary of file paths (keys), and 'found indicators' (boolean values).
            Returns:
                list: a list of strings, representing the paths of files that have NOT been found. 100 percent success == empty list.
            """
            return_list = []

            for file_path in file_dict:
                if not file_dict[file_path]:
                    return_list.append(file_path)

            return return_list

        # Prepare a data structure for the files to check for.
        # Essentially, we convert the list of include files to a dictionary, with the value being a boolean.
        # The boolean indicates whether the file has been successfully found or not.
        file_dict = prepare_data_structure()

        # Now iterate through the provided root paths, appending the sample files to a given root path and check if it exists.
        # By the end of this process, hopefully all values in file_checks will be set to True (indicating they have been found).
        for root_path in paths_list:
            for include_path in file_dict:
                was_found = file_dict[include_path]

                if not was_found:
                    # The file has not yet been located, so we check if it exists in root_path.
                    joined_path = os.path.join(root_path, include_path)

                    if os.path.isfile(joined_path):
                        # Yes! the file exists; switch False to True.
                        file_dict[include_path] = True

        # Now check if all the files were found.
        failed_files = get_missing_files(file_dict)

        if len(failed_files) > 0:
            fail_message = ["The installer couldn't locate one or more header files. These are listed below.",
                            "Please ensure that you are specifying the correct location to the required files.",]

            for filename in failed_files:
                fail_message.append("    * {filename}".format(filename=filename))

            fail_message.append("We cannot compile esig without access to these files.")

            message_printer(fail_message, is_failure=True, terminate=True)

        message_printer("Include files successfully found.")


class BinaryDistribution(Distribution):
    """
    A simple class that tells setuptools that the package is a binary distribution -- that is, it is not "pure" Python.
    """
    def is_pure(self):
        """
        Always returns False, denoting that the package is not "pure" Python.

        Args:
            None

        Returns:
            bool: False
        """
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


class BuildExtensionCommand(build_ext):
    """
    Extends the build_ext class, allowing for the injection of additional code into the build_ext process.
    """
    def run(self):
        """
        Attempts to import numpy and append the result of numpy.get_include() to the self.include_dirs list.
        This is to avoid the circular issue of importing numpy at the top of the module.
        See https://stackoverflow.com/a/42163080.

        Args:
            self (NumpyExtensionCommand): Instance of self.

        Returns:
            None
        """
        message_printer("Running extra esig pre-build commands...")

        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


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
    def __init__(self):
        message_printer("Starting esig installer...")
        self.__package_abs_root = None
        self.__include_dirs = None
        self.__library_dirs = None

        self.__run_checks()


    def __run_checks(self):
        """
        Runs a series of startup checks. At present:
            - checks the version of Python;
            - checks for required libraries; and
            - checks for required header (include) files.
        If all checks pass, then compilation should work.

        Args:
            self (InstallationConfiguration): instance of InstanceConfiguration.

        Returns:
            None
        """
        if not ComponentChecker.check_python_version():
            error_message = ["Your version of Python is incompatible with esig.",
                             "At a minimum, you require version {version}. Please consider upgrading and try again.".format(
                                 version='.'.join(str(x) for x in MINIMUM_PYTHON_VERSION))]

            message_printer(error_message, is_failure=True, terminate=True)

        if sys.argv[1].lower() == 'install':
            for argument in sys.argv:
                if argument.startswith('--include-dirs='):
                    self.include_dirs = argument[15:] # use length '--include-dirs=' instead

                if argument.startswith('--library-dirs='):
                    self.library_dirs = argument[15:] # use length '--library-dirs=' instead

            ComponentChecker.check_libraries(self.library_dirs)
            ComponentChecker.check_includes(self.include_dirs)


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
    def package_abs_root(self):
        """
        Returns the absolute root path of the esig package.
        If not aleady set, None is returned.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
        Returns:
            string: a string representation of the path; None if not currently set.
        """
        return self.__package_abs_root


    @package_abs_root.setter
    def package_abs_root(self, path):
        """
        Sets the absolute root path of the esig package.

        Args:
            self (InstallationConfiguration): an object instance of InstanceConfiguration
            path (string): the path, represented as string
        Returns:
            None
        """
        self.__package_abs_root = path


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



# A series of constants that are provided by the module.
MINIMUM_PYTHON_VERSION = (2,7)  #NOTE: this should be changed for the new version
MESSAGE_PREFIX = 'esig_install> '  # Prefix appended to every message displayed by this module.
PLATFORMS = Enum(['WINDOWS', 'LINUX', 'MACOS', 'OTHER'])

# singleton
CONFIGURATION = InstallationConfiguration()
