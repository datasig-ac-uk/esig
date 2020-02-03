#!/usr/bin/env python

#
# The ESig Python package
# Basic package functions
#


import os


__author__ = 'David Maxwell <dmaxwell@turing.ac.uk>'
__date__ = '2017-07-21'


ESIG_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


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
    version_string = f.read().split(' ')
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
        