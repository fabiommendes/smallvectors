=======
Vectors
=======

As the name suggest, :mod:`smallvectors` is a linear algebra library for
manipulating vectors (and other objects) of low dimensionality. The most basic
type is a vector:

>>> from smallvectors import Vec
>>> v = Vec(1, 2)
>>> u = Vec(3, 4)
>>> u + v
Vec(4, 6)

As with most elements in the :mod:`smallvectors` package, vectors have the
default immutable, and an mutable implementation. Both derive from a VecAny
base class:

>>> from smallvectors import Vec, VecAny, mVec
>>> isinstance(u, mVec)
False
>>> isinstance(v, VecAny)
True

Vectors classes are also parametrized by type and dimension. Hence we have an
infinite type hierarchy for each scalar type. We use the rectangular bracket
notation in order to designate the correct vector type

>>> Vec[2, int](1.5, 2)
Vec(1, 2)


Direction
=========

:mod:`smallvectors` also define a Direction subclass for Vec. These elements
represent unity vectors. A separate subclass is handy in situations in which
normalization is required for some variable or attribute.

There is no mutable Direction type since most operations that can be done in a
unity vector spoils the normalization constraint.

Points
======

:mod:`smallvectors` also defines a separate Point type. While vectors are
quantities with magnitude and direction, Point objects represent directly
points in space.