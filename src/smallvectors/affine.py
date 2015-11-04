'''
======================
Affine transformations
======================

Affine objects expose the matrix and displacement vector parts in the
``affine.matrix`` and ``affine.vector`` attributes.


Arithmetic operations
=====================

Affine transformation class define arithmetical mathematical operations
that can be useful to construct different affine transformations. These
operations do not define an algebra in the strict mathmatical sense but are
still useful to have (for those mathematically inclined, technically it is
a ...?).

It is simpler to explain by examples. The affine transformation is
presented as a matrix augmented by a column vector

>>> A = Affine([[1, 2], [3, 4]], [5, 6])
>>> print(A)
|1  2 : 5|
|3  4 : 6|

The terms 1, 2, 3, 4 at the left represent the matrix part of the affine
transformation and can be accessed in the matrix attribute. The column in
the right (5, 6) represents the translation vector.

This transformation can be applied to a vector using either the ``A * v``
or the ``A(v)`` notations. The result is equivalent to multiply ``v`` by
``A.linear`` part and then add ``A.translation``.

>>> A(1, 0)  # same as A((1, 0)) or A * (1, 0)
Vec(6, 9)

Of course this result is the same as applying the transformation manually

>>> v = Vec(1, 0)
>>> A.A * v + A.b
Vec(6, 9)

Multiplication by a vector (to the right) is the same as function
application. There are however other operations that create new affine
transformations. Adding or subtracting vectors return a new affine
transformation in which only the vector part is modified. Geometrically
this can be interpreted as a new transformation that translates the result
of the previous transformation. More simply: if $A$ takes a vector, do some
linear transforms and a translation afterwards, $(A + b)$ will perform the
same operations and translate the result by b at the end.

>>> print(A - (4, 4))
|1  2 : 1|
|3  4 : 2|

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
rotates the vector. It is up to the reader to work out some sittuations in
which the order of transformations is clearly important.


Representations
===============

We are viewing an affine transformation as combination of a linear
transformation (matrix) plus a translation (vector). They can be
represented differently. First, if we are only concerned about the
coordinates of the transform, we can obtain it as a matrix with the
translation vector appended to the rightmost column.

>>> print(A.asmatrix())
|1  2  5|
|3  4  6|

This has some incoveniences. The 2x3 matrix cannot be used to linearly
transform vectors since it does not even have the correct shape. A matrix
as shown above requires 3D vectors instead of the 2D that the affine
transform operates. We can however use the following trick: take a (x, y)
bidimensional vector and append a coordinate with value 1 resulting in
(x, y, 1). When we apply the above matrix to that vector the result is the
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


We can also construct a affine transform from these elements using the
proper constructors

>>> T = Affine.frommatrix([[1, 2, 5],
...                        [3, 4, 6]])
>>> A == T
True

As usual, the affine transformation object can be flattened by the
``affine.flat`` attribute

>>> list(A.flat)
[1, 2, 5, 3, 4, 6]

'''

import numbers
from smallvectors.core import SmallVectorsBase, AddElementWise
from smallvectors import Mat, Vec, asvector, asmatrix
from smallvectors.tools import flatten, dtype


class Affine(SmallVectorsBase, AddElementWise):
    '''
    A generic affine transformation. Affine transformations are a combination
    of linear transformation + translation. A affine operation $A(v)$ in a
    vector $v$ can be described as

    math::
        A(v) = M v + b,

    for some arbitrary matrix M and displacement vector b. Mathematicians often
    like to define it as a transformation that maps line to lines and keep them
    parallel.
    '''
    
    __parameters__ = (int, type)
    
    @classmethod
    def __preparenamespace__(cls, params):
        N, _dtype = params
        ns = SmallVectorsBase.__preparenamespace__(params)
        ns['size'] = N * (N + 1)
        return ns
    
    @staticmethod
    def __finalizetype__(cls):
        SmallVectorsBase.__finalizetype__(cls)
        N = cls.shape[0]
        if isinstance(N, int):
            cls.size = N * (N + 1)
        cls.dim = N
        
    @classmethod
    def __abstract_new__(cls, matrix, vector):
        data, N, M = flatten(matrix)
        if N != M:
            raise ValueError('expected an square matrix')
        
        M += 1
        for i, x in enumerate(vector):
            data.insert((i + 1) * M - 1, x)
        T = cls[N, dtype(data)]
        return T.fromflat(data)
    
    def __init__(self, matrix, vector):
        # Convert values
        if isinstance(matrix, Mat):
            matrix = matrix
        else:
            matrix = Mat(*matrix)

        # linearize inputs and save in the flat attribute
        self.shape = matrix.shape
        data = list(map(list, matrix))
        for x, L in zip(vector, data):
            L.append(x)
        self.flat = sum(data, [])

    def __repr__(self):
        name = self.__origin__.__name__
        A, b = self
        data = map(repr, [[list(r) for r in A], list(b)])
        return '%s(%s)' % (name, ', '.join(data))
    
    def __str__(self):
        lines = str(self.asmatrix()).splitlines()
        for i, line in list(enumerate(lines)):
            pre, _, pos = line.rpartition(' ')
            lines[i] = (pre.strip(), pos.lstrip())
        
        # Apply maximum alignment
        alignpre = max([len(x) for x, _ in lines])
        alignpos = max([len(x) for _, x in lines])
        for i, (pre, post) in list(enumerate(lines)):
            lines[i] = pre.ljust(alignpre) + ' : ' + post.rjust(alignpos)
        return '\n'.join(lines)            

    @classmethod
    def frommatrix(cls, matrix):
        '''Construct a new affine transformation from a matrix
        representation'''

        matrix, vector = asmatrix(matrix).droppingcol()
        return cls(matrix, vector)

    @property
    def linear(self):
        '''Linear part of the transformation'''

        N = self.dim
        data = list(self.flat)
        del data[N::N + 1]
        return Mat[N, N, self.dtype].fromflat(data)

    @property
    def translation(self):
        '''Translational part of the transformation'''

        N = self.dim
        return Vec[N, self.dtype].fromflat(self.flat[N::N + 1], copy=False)

    A, b = linear, translation

    #
    # Data representation
    #
    def asmatrix(self):
        '''Return a minimum matricial representation of the affine
        transformation'''

        return self.A.withcol(self.b)

    def asaugmented(self):
        '''Return the augmented matrix representation of the affine transform.
        The augmented matrix operates linearly in one dimension greater than
        the original transform. The trick is to append 1 to the last component
        of a vector in order to mimick the same results of the affine
        transformation using strictly linear operations.'''

        bottom = [0] * (self.dim + 1)
        bottom[-1] = 1
        return self.asmatrix().withrow(bottom)

    # Behaves somewhat as a tuple of (matrix, vector). Useful for extracting
    # both of these components in the same line.
    def __iter__(self):
        yield self.A
        yield self.b

    def __getitem__(self, idx):
        if idx in (0, -2):
            return self.matrix
        elif idx in (1, -1):
            return self.vector
        else:
            return super().__getitem__(idx)

    def __len__(self):
        return 2

    # Call delegates to multiplication with the proper object
    def __call__(self, *args):
        if len(args) > 1:
            return self * Vec(*args)
        else:
            return self * args[0]

    #
    # Arithmetic operations
    #
    def __mul__(self, other):
        if isinstance(other, Vec):
            return self.A * other + self.b
        elif isinstance(other, Affine):
            return Affine(self.A * other.A, self.b + self.A * other.b)
        elif isinstance(other, (Mat, numbers.Number)):
            return Affine(self.A * other, self.b)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Vec):
            return other * self.A + self.b
        elif isinstance(other, (Mat, numbers.Number)):
            return Affine(other * self.A, other * self.b)
        else:
            return NotImplemented

    def __sub__(self, other):
        return Affine(self.A, self.b - other)

    def __add__(self, other):
        return Affine(self.A, self.b + other)


class Similarity(Affine):

    '''
    A similarity transform is a special kind of affine transform that keeps the
    shape of objects invariant. It can change location, scale, rotation and
    perform reflections. Shear transformations are forbidden since the alter
    shape.

    Not only it can be easier to reason about than generic affine
    transformation, a separeate Similarity type can be be more efficient and
    stable in some circunstances. Take for example an affine transform in
    a circle: circles can be transforemed into other circles or into ellipsis.
    Given some arbitrary affine transform, it is hard to figure out if the
    result can still be represented as a circle or if an ellipse will result.
    It is easier to reason if we have a higher level view of the affine
    transform: instead of specifing the components directly, we talk about
    rotations, translations, scaling, and reflections. By the way, all these
    operations keeps the "circleness" of a circle intact, which responds the
    original question.

    Parameters
    ----------
    angle : float
        Angle (in degrees) of rotation.

    '''

    def __init__(self, angle=None, delta=(0, 0), delta_before=(0, 0),
                 scale=1, flip_x=False, flip_y=False, theta=None):

        self.theta = theta
        self.delta = delta
        self.scale = scale
        self.flip_x = flip_x
        self.flip_y = flip_y

    def __call__(self, *args):
        if len(args) > 1:
            return self * args
        else:
            obj = args[0]
            if isinstance(obj, (Vec, tuple, list)):
                return self * obj
            try:
                return obj.similar_transform(self)
            except AttributeError:
                return obj.affine_transform(self)


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)

    T1 = Affine([[1, 2], [0, 1]], [3, -4])
    T2 = Affine([[1, 2], [0, 1]], [0, 0])
    M2 = T2.A
    v = Vec(1, 2)
    print(T1 * T2)
    print(T1 * M2)
    print((T1 * T2) * v)
    print(T1 * (T2 * v))
    print(T1 * (M2 * v))