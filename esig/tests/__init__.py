import sys
import unittest
from . import test_package


def get_suite():
    """
    Constructs and returns a Python unittest suite object.
    This can be used to run the unit tests for ESig.
    
    Args:
        None
    Returns:
        suite: A Python unittest suite, referring to all tests specified within the tests package.
    """
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(test_package.TestESIG))
    
    return suite


def run_tests(terminate=False):
    """
    Acquires the ESig test suite, and runs the tests.
    
    Args:
        terminate (bool): if set to True, will terminate on failure; otherwise, returns False.
    Returns:
        bool: True iif all tests pass successfully; False otherwise.
    """
    suite = get_suite()
    runner = unittest.TextTestRunner()
    status = runner.run(suite).wasSuccessful()

    if not status and terminate:
        sys.exit(1)