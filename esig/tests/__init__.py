import sys
import os
import unittest
from . import test_package


def run_tests(terminate=False): # argument ignored
    """
    Acquires the ESig test suite, and runs the tests.

    Returns:
        terminates with exit code 1 if all tests pass successfully; 0 otherwise.
    """
    package_root = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.TestLoader().discover(package_root)
    print(suite.countTestCases() + " test cases found.")
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    print(result)

    if not result.wasSuccessful():
        sys.exit(1)
    else:
        sys.exit(0)
