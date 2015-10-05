'''
Loads all tests in package and run
'''

from smallvectors.tests import *  # @UnusedWildImport
import smallvectors as mod_current

try:
    from unittest2 import main
except ImportError:
    from unittest import main
import doctest
import sys


def load_tests(loader, tests, ignore):
    prefix = mod_current.__name__

    # Find doctests
    for modname, mod in sys.modules.items():
        if modname.startswith(prefix + '.') or modname == prefix:
            try:
                tests.addTests(doctest.DocTestSuite(mod))
            except ValueError:  # no docstring
                pass

    return tests

main()
