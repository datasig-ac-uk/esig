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
    suite.addTest(unittest.makeSuite(test_package.TestRecombine))

    return suite


def run_tests(terminate=False): # argument ignored
    """
    Acquires the ESig test suite, and runs the tests.

    Returns:
        terminates with exit code 1 if all tests pass successfully; 0 otherwise.
    """
    suite = get_suite()
    runner = unittest.TextTestRunner()
    status = runner.run(suite).wasSuccessful()

    if not status:
        sys.exit(1)
    else:
        sys.exit(0)
