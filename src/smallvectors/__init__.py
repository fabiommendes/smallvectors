'''
============
Smallvectors
============

The ``smallvectors`` library provides efficient implementations for vectors,
matrices and other linear algebra constructs of small dimensionality. This code
is useable and stable, but it is still beta and many classes may be not fully
implemented or may not be optimized yet.

Examples
========

>>> u = Vec(1, 2)
>>> v = Vec(3.0, 4)
>>> u + v
Vec(4.0, 6.0)

'''

from .meta import __version__, __author__
from .core import *
from .sequence import seq
from .vector import *
from .direction import *
from .matrix import *
from .affine import *
from .rotmat import *
from .tools import *
from .vecarray import *
from .overloads import *