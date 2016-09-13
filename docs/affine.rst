======================
Affine transformations
======================

.. code:: python
   :class: hidden

   from smallvectors import *

Affine transformations are a combination of a displacement with a linear
transformation.  Affine objects expose the matrix and displacement vector parts
respectively in the ``affine.linear`` and ``affine.translation`` attributes.

We have to pass these two parameters to construct an Affine object

>>> M = [[1, 2],
...      [3, 4]]
>>> A = Affine(M, [5, 6])
>>> print(A)
|1  2 : 5|
|3  4 : 6|


We apply affine transforms to vectors using either the ``A * v`` or the ``A(v)``
notations. The result is equivalent to multiply ``v`` by ``A.linear`` part and
then add ``A.translation``.

>>> A(1, 0)  # same as A((1, 0)) or A * (1, 0)
Vec(6, 9)

Of course, this result is the same as applying the transformation manually

>>> v = Vec(1, 0)
>>> A.linear * v + A.translation
Vec(6, 9)

Multiplication by a vector (to the right) is the same as function
application. There are however other operations that create new affine
transformations. Adding or subtracting vectors return a new affine
transformation in which only the vector part is modified. Geometrically
this can be interpreted as a new transformation that translates the result
of the previous transformation.

>>> print(A - A.translation)
|1  2 : 0|
|3  4 : 0|

We can multiply the affine transformation by a number or matrix in order to
affect both the matrix and vector parts.

>>> print(2 * A)
|2  4 : 10|
|6  8 : 12|

Some care must be given to the order of operation of matrix multiplication.
It is not commutative, but it behaves as expected.

>>> M = Mat([0, 1], [1, 0])
>>> print(A * M)
|2  1 : 5|
|4  3 : 6|
>>> print(M * A)
|3  4 : 6|
|1  2 : 5|

The geometrical interpretation depends on what the matrix M do. If M is a
rotation, then A * M rotates a vector and then apply the original affine
transformation while M * A applies the affine transformation and then
rotates the vector.

Representations
===============

We are viewing an affine transformation as combination of a linear
transformation (matrix) plus a translation (vector). They can be
represented differently. First, if we are only concerned about the
coordinates of the transform, we can obtain them as a matrix with the
translation vector appended to the rightmost column.

>>> print(A.as_matrix())
|1  2  5|
|3  4  6|

This has some inconveniences. The 2x3 matrix cannot be used to linearly
transform 2D vectors since it does not even have the correct shape. A matrix
as shown above requires 3D vectors instead of the 2D that the affine
transform operates. We can however use the following trick: take a
bidimensional vector ``(x, y)`` and append a coordinate with value 1 resulting in
``(x, y, 1)``. When we apply the above matrix to that vector the result is the
identical to the affine transformation!

If we want to preserve this extra component, the trick is to append a new
row to obtain a square matrix. Now we have a linear transform in three
dimensions in which the first two components behave as an affine
transformation. This augmented matrix can be obtained by the following
method.

>>> print(A.asaugmented())
|1  2  5|
|3  4  6|
|0  0  1|


We can also construct affine transforms from these elements using the proper
constructors

>>> T = Affine.frommatrix([[1, 2, 5],
...                        [3, 4, 6]])
>>> A == T
True

As usual, the affine transformation object can be flattened by the
``affine.flat`` attribute

>>> list(A.flat)
[1, 2, 5, 3, 4, 6]

Class reference
---------------

.. automodule:: smallvectors.affine
   :members:
   :inherited-members: