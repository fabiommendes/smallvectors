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
with open(os.path.join(dirname, 'src', project, '__meta__.py'), 'w', encoding='utf-8') as F:
    F.write('#-*- coding: utf-8 -*-\n'
            '__version__ = %r\n'
            '__author__ = %r\n' % (version, author))

# Chooses the default Python3 branch or the code converted by 3to2
PYSRC = 'src' if sys.version_info[0] == 3 else 'py2src'

# Cython stuff (for the future)
setup_kwds = {}
if 'PyPy' not in sys.version and False:
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
    url='http://github.com/fabiommendes/smallvectors/',
    description='Linear algebra objects for small dimensions.',
    long_description=open('README.rst').read(),

    # Classifiers (see https://pypi.python.org/pypi?%3Aaction=list_classifiers)
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries',
    ],

    # Packages and depencies
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=[
        'six',
        'pygeneric>=0.5.4',
        'lazyutils',
    ],
    extras_require={
        'dev': [
            'pytest',
            'manuel',
            'sphinx',
        ],
    },

    # Other configurations
    zip_safe=False,
    platforms='any',
    **setup_kwds
)