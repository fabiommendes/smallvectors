#-*- coding: utf8 -*-
import os
import setuptools
import sys
from setuptools import setup

#
# Read VERSION from file and write it in the appropriate places
#
AUTHOR = 'Fábio Macêdo Mendes'
BASE, _ = os.path.split(__file__)
with open(os.path.join(BASE, 'VERSION')) as F:
    VERSION = F.read().strip()
path = os.path.join(BASE, 'src', 'smallvectors', 'meta.py')
with open(path, 'w') as F:
    F.write(
        '# Auto-generated file. Please do not edit\n'
        '__version__ = %r\n' % VERSION +
        '__author__ = %r\n' % AUTHOR)

#
# Choose the default Python3 branch or the code converted by 3to2
#
PYSRC = 'src' if sys.version.startswith('3') else 'py2src'

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
    long_description=open(os.path.join(BASE, 'README.txt')).read(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries',
    ],

    package_dir={'': PYSRC},
    packages=setuptools.find_packages(PYSRC),
    license='GPL',
    install_requires=['six', 'pygeneric>=0.1.1'],
)
