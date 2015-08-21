'''
============
Smallvectors
============

The ``smallvectors`` library provides efficient implementations for vectors,
matrices and other linear algebra constructs of small dimensionality. This code
is useable and stable, but it is still beta and many classes may be not fully
implemented or may not be optimized yet.
'''

from .sequence import seq
from .cartesian import *
from .mat2 import *  # FIXME
from .tools import *
from .vecarray import *
