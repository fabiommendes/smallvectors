'''
===============================================================
Base classes for mathematical types in the smallvectors package
===============================================================


Take the vector type, for instance. Ideally we want to have the following class
structure::

    AnyVec
     |-- Vec
     |    |-- Vec[1]: immutable types for specific dimensions
     |    |-- Vec[2]
     |    |    ...
     |    |-- Vec[N]
     |    |    \-- Direction[N]: unitary vectors
     |    \-- Vec[N, T]: immutable types with components of type T
     |         \-- Direction[N, T]: unitary vectors of type T
     \-- mVec
          |-- mVec[N]: mutable vector type
          \-- mVec[N, T]: mutable type with type T components.

This structure obbeys Liskov principle: a property that can be computed to any
vector, such as computing its norm, obviously can be computed to, say, 2
dimensional directions. However, it is not the whole story. It is easy to
conceive properties for arbitrary Vec types: e.g., we can safely share
immutable vectors in different threads, so we want ``isinstance(v, Vec)`` to
work as expected, hence Vec[N] must inherit from both and abstract Vec class
and from the concrete gVec[N].

From an implementation level, however, things are not as simple. Vector may
share a lot of implementation with Point, although both are in a completely
separate class hierarchy. It may even share some implementation with other
mathematical objects such as matrices, quaternions, etc. Also, we want to be
able to plug partial implementations to each type in order to optimize for
various Python platforms. There is a generic N-dimensional implementation that
is a reference for correctness, and some pluggable partial implementations that
optimize for specific platforms: pure Python for PyPy, IronPython and Jython
and a Cython based one for CPython (which can fallback to pure Python if the
user does not have a C compiler).

All these implementations talk to each other via the ``object.flat`` attribute,
which must be iterable and indexable. Base implementations should only provide
this attribute with the proper semantics and everything should work as
expected. For instance, we may want to store the coordinates of a 2D vector in
the vector.x and vector.y attributes. The only thing the base class must
provide for the rest of the api to work is a list-like view of these attributes
as ``obj.flat ~ [x, y]``.

Usage
=====

>>> u = Vec(1, 2)
>>> v = Vec(3, 4)
>>> u + v
Vec(4, 6)
'''
