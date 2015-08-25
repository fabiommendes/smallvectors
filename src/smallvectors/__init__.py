'''
============
Smallvectors
============

The ``smallvectors`` library provides efficient implementations for vectors,
matrices and other linear algebra constructs of small dimensionality. This code
is useable and stable, but it is still beta and many classes may be not fully
implemented or may not be optimized yet.
'''

from .meta import __version__, __author__
from .sequence import seq
from .cartesian import *
from .matrix import *
from .tools import *
from .vecarray import *
