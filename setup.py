#-*- coding: utf8 -*-
import setuplib
from distutils.core import setup

VERSION = '0.1a'
setup_kwds = {}


#
# Main configuration script
#
setup(
    name='smallvectors',
    version=VERSION,
    description='Efficient linear algebra in low dimensions',
    author='Fábio Macêdo Mendes',
    author_email='fabiomacedomendes@gmail.com',
    url='https://github.com/fabiommendes/smallshapes',
    long_description=(
        r'''A lightweight library that implements linear algebra operations
in low dimensions. These objects were create to be used in a game engine, but
may be useful elsewhere.

Includes:
    * Vector, point and direction types
    * Arbitrary shape matrix types and some specialized matrices
    * Affine transforms
    * Quaternions
    '''),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries',
    ],

    package_dir={'': 'src'},
    packages=setuplib.get_packages('src'),
    license='GPL',
    requires=['six'],
)
