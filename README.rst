.. image:: https://travis-ci.org/fabiommendes/smallvectors.svg?branch=master
    :target: https://travis-ci.org/fabiommendes/smallvectors

.. image:: https://coveralls.io/repos/github/fabiommendes/smallvectors/badge.svg?branch=master
    :target: https://coveralls.io/github/fabiommendes/smallvectors?branch=master


`smallvectors` is a lightweight library that implements linear algebra operations
in low dimensions. These objects were create to be used in a game engine, but
may be useful elsewhere.

The library has an opt-in process for optimizing code with C-extensions written
in Cython. The optimized version of the library is (or will be) one of the
fastest Python packages for low dimensional linear algebra operations.

Small vectors supports:

* Vector, point and direction types
* Arbitrary matrices and some specialized ones (square matrices, unitary
  matrices, rotational matrices, etc).
* Affine transforms
* Quaternions
