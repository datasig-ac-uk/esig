#!/usr/bin/env python

#
# The ESig Python package
# Basic package functions
#


import os


__author__ = 'David Maxwell <dmaxwell@turing.ac.uk>'
__date__ = '2017-07-21'


ESIG_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Python 3.8+ doesn't use PATH to locate DLLs; need to use add_dll_directory instead.
# Not sure if this is the right place to do this, though; 3.8 Porting Guide says to do
# this "while loading your library" (https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew).
# Note that add_dll_directory doesn't exist on non-Windows platforms or before Python 3.8.
import sys
try:
    from os.path import expanduser
    #recombine_dll_dir = os.path.join(expanduser("~"), "lyonstech", "bin")
    recombine_dll_dir = ESIG_PACKAGE_ROOT
    os.add_dll_directory(recombine_dll_dir)
except AttributeError:
    pass
    #print("Ignoring attempt to add_dll_directory.")

def get_version():
    """
    Returns the version number of the ESig package.
    The version is obtained from the VERSION file in the root of the package.

    Args:
        None

    Returns:
        string: The package version number. In format 'major.minor.release'.
    """
    version_filename = os.path.join(ESIG_PACKAGE_ROOT, 'VERSION')
    f = open(version_filename, 'r')
    version_string = f.read().strip().split(' ')
    f.close()

    return '.'.join(version_string)

def is_library_loaded():
    """
    Determines whether the tosig Python extension can be successfully loaded.
    If the library cannot be loaded successfully, debugging information can be obtained from get_library_load_error().

    Args:
        None
    Returns:
        boolean: True iif the library can be loaded successfully; False otherwise.
    """
    try:
        from esig import tosig
    except ImportError:
        return False

    return True


def get_library_load_error():
    """
    Returns a string containing the message of the exception raised when attempting to import tosig.
    If no exception is raised when attempting to import, None is returned.

    Args:
        None
    Returns:
        string: The message associated with the exception when attempting to import.
        None: If no exception is raised when importing, None is returned.
    """
    try:
        from esig import tosig
        return None
    except ImportError as e:
        return e.msg
