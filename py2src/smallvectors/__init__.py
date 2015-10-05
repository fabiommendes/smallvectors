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
Vec[2, float](4.0, 6.0)

'''

from .meta import __version__, __author__
from .sequence import seq
from .vec_or_point import *
from .matrix import *
from .tools import *
from .vecarray import *
