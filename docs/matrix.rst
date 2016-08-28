========
Matrices
========

Matrices are most often created from a list of lists

>>> M = Mat([1, 2],
...         [3, 4])


Matrix mutiplication is done by the ``*`` operator, which works as in linear
algebra:

>>> print(M * M)
| 7  10|
|15  22|

Matrices can also interact with vectors, however one should be aware of some
caveats. The "column vectors" and "row vectors" of linear algebra are, in fact,
just n x 1 or 1 x n matrices. :mod:`smallshapes`'s Vec class are abstract
vectors of n components and make no mention to its layout in matrix form.

Vector with matrix multiplications are treated "optimistically": if operation
is valid if ``v`` is a row vector, it is treated as a row vector (and likewise
for column vectors). The result is always a vector.

>>> v = Vec(1, 2)
>>> matrix * v
Vec(5, 11)
>>> v * matrix
Vec(7, 10)

