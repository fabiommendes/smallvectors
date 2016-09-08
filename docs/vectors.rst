Vectors
=======

.. code:: python
   :class: hidden

   from smallvectors import *

As the name suggest, :mod:`smallvectors` is a linear algebra library for
manipulating vectors (and other objects) of low dimensionality. The most basic
type is a vector:

>>> v = Vec(1, 2)
>>> u = Vec(3, 4)
>>> u + v
Vec(4, 6)

As with most elements in the :mod:`smallvectors` package, vectors have the
default immutable, and an mutable implementation. Both derive from a VecAny
base class:

>>> isinstance(u, mVec)
False
>>> isinstance(v, VecAny)
True

Vectors classes are also parametrized by type and dimension. Hence we have an
infinite type hierarchy for each scalar type. We use the rectangular bracket
notation in order to designate the correct vector type

>>> Vec[2, int](1.0, 2)
Vec(1, 2)
>>> Vec[2, float](1, 2)
Vec(1.0, 2.0)



Class reference
---------------

.. autoclass:: smallvectors.Vec
   :members:
   :inherited-members:

.. autoclass:: smallvectors.mVec
   :members:


Useful functions
----------------

.. automodule:: smallvectors.vector
    :members: asvector, asmvector, asavector

Direction
=========

:mod:`smallvectors` also define a Direction subclass for Vec. These elements
represent unity vectors. A separate subclass is handy in situations in which
normalization is required for some variable or attribute.

There is no mutable Direction type since most operations that can be done in a
unity vector spoils the normalization constraint.


Class reference
---------------

.. autoclass:: smallvectors.Direction
   :members:

Useful functions
----------------

.. automodule:: smallvectors.vector
   :members: asdirection

Points
======

:mod:`smallvectors` also defines a separate Point type. While vectors are
quantities with magnitude and direction, Point objects represent directly
points in space.


Class reference
---------------

.. autoclass:: smallvectors.Point
   :members:

.. autoclass:: smallvectors.mPoint
   :members:


Useful functions
----------------

.. automodule:: smallvectors.vector
   :members: aspoint, asmpoint, asapoint
