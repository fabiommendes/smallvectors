# -*- coding: utf8 -*-
import os

import sys
from setuptools import setup, find_packages


# Meta information
name = 'smallvectors'
project = 'smallvectors'
author = 'Fábio Macêdo Mendes'
version = open('VERSION').read().strip()
dirname = os.path.dirname(__file__)


# Save version and author to __meta__.py
with open(os.path.join(dirname, 'src', project, '__meta__.py'), 'w') as F:
    F.write('__version__ = %r\n__author__ = %r\n' % (version, author))

# Chooses the default Python3 branch or the code converted by 3to2
PYSRC = 'src' if sys.version_info[0] == 3 else 'py2src'

# Cython stuff (for the future)
setup_kwds = {}
if 'PyPy' not in sys.version:
    try:
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext
    except ImportError:
        import warnings
        warnings.warn(
            'Please install Cython to compile faster versions of the '
            'smallvectors package',
        )
    else:
        try:
            setup_kwds.update(
                ext_modules=cythonize('src/smallvectors/*.pyx'),
                cmdclass={'build_ext': build_ext})
        except ValueError:
            pass


setup(
    # Basic info
    name=name,
    version=version,
    author=author,
    author_email='fabiomacedomendes@gmail.com',
    url='',
    description='A short description for your project.',
    long_description=open('README.rst').read(),

    # Classifiers (see https://pypi.python.org/pypi?%3Aaction=list_classifiers)
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries',
    ],

    # Packages and depencies
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=[
        'six',
        'pygeneric',
        'lazytools',
    ],
    extras_require={
        'testing': ['pytest', 'manuel'],
    },

    # Other configurations
    zip_safe=False,
    platforms='any',
    test_suite='%s.test.test_%s' % (name, name),
    **setup_kwds,
)