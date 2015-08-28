#-*- coding: utf8 -*-
import os
import setuptools
from setuptools import setup

VERSION = '0.3a0'
AUTHOR = 'Fábio Macêdo Mendes'

#
# Create meta.py file with updated version/author info
#
base, _ = os.path.split(__file__)
path = os.path.join(base, 'src', 'smallvectors', 'meta.py')
with open(path, 'w') as F:
    F.write(
        '# Auto-generated file. Please do not edit\n'
        '__version__ = %r\n' % VERSION +
        '__author__ = %r\n' % AUTHOR)


#
# Main configuration script
#
setup(
    name='smallvectors',
    version=VERSION,
    description='Efficient linear algebra in low dimensions',
    author=AUTHOR,
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
    packages=setuptools.find_packages(),
    license='GPL',
    requires=['six', 'pygeneric'],
)
