# -*- coding: utf8 -*-
'''
Affine operations
'''

from smallvectors import Matrix, vector, asvector, Vec, Vec2


class Affine(object):

    '''
    A generic affine transformation. Affine transformations are a combination
    of linear transformation + translation. A affine operation $A(v)$ in a
    vector $v$ can be described as

    math::
        A(v) = M v + b,

    for some arbitrary matrix M and displacement vector b. Mathematicians often
    like to define it as a transformation that maps line to lines and keep them
    parallel.

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

    >>> A = Affine([[1, 2], [3, 4]], [5, 6]); A
    |1  2 : 5|
    |3  4 : 6|

    The terms 1, 2, 3, 4 at the left represent the matrix part of the affine
    transformation and can be accessed in the matrix attribute. The column in
    the right (5, 6) represents the translation vector.

    This transformation can be applied to a vector using either the ``A * v``
    or the ``A(v)`` notations. The result is equivalent to multiply ``v`` by
    ``A.matrix`` part and then add ``A.vector``.

    >>> A(1, 0)  # same as A((1, 0)) or A * (1, 0)
    Vec2(6, 9)

    Of course this result is the same as applying the transformation manually

    >>> v = vector(1, 0)
    >>> A.matrix * v + A.vector
    Vec2(6, 9)

    Multiplication by a vector (to the right) is the same as function
    application. There are however other operations that create new affine
    transformations. Adding or subtracting vectors return a new affine
    transformation in which only the vector part is modified. Geometrically
    this can be interpreted as a new transformation that translates the result
    of the previous transformation. More simply: if $A$ takes a vector, do some
    linear transforms and a translation afterwards, $(A + b)$ will perform the
    same operations and translate the result by b at the end.

    >>> A - (4, 4)
    |1  2 : 1|
    |3  4 : 2|

    We can multiply the affine transformation by a number or matrix in order to
    affect both the matrix and vector parts.

    >>> 2 * A
    |2  4 : 10|
    |6  8 : 12|

    Some care must be given to the order of operation of matrix multiplication.
    It is not commutative, but it behaves as expected.

    >>> M = Matrix([[0, 1], [1, 0]])
    >>> A * M
    |2  1 : 6|
    |4  3 : 5|
    >>> M * A
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

    >>> A.as_matrix()
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

    >>> A.as_augmented_matrix()
    |1  2  5|
    |3  4  6|
    |0  0  1|


    We can also construct a affine transform from these elements using the
    proper constructors

    >>> T = Affine.from_matrix([[1, 2, 5],
                ...             [3, 4, 6]])
    >>> A == T
    True

    As usual, the affine transformation object can be flattened by the
    ``affine.flat`` attribute

    >>> list(A.flat)
    [1, 2, 5, 3, 4, 6]

    '''

    def __init__(self, matrix, vector):
        # Convert values
        if isinstance(matrix, Matrix):
            matrix = matrix
        else:
            matrix = Matrix(matrix)

        # linearize inputs and save in the flat attribute
        self.shape = matrix.shape
        data = matrix.as_lists()
        for x, L in zip(vector, data):
            L.append(x)
        self.flat = sum(data, [])

    @classmethod
    def from_matrix(cls, matrix):
        '''Construct a new affine transformation from a matrix
        representation'''

        M = Matrix(matrix)
        matrix, vector = M.drop_col()
        return cls(matrix, vector)

    @classmethod
    def from_flat(cls, flat, N, M):
        '''Construct a new affine transformation from the flat data and the
        shape of the matrix part of the transform.'''

        new = object.__new__(cls)
        new.shape = (N, M)
        new.flat = list(flat)
        return new

    @property
    def matrix(self):
        '''Matrix part of the transformation'''

        N, M = self.shape
        size = N * M
        flat = [0] * (size)
        data = self.flat
        for i in range(size):
            flat[i] = data[i + i // M]

        return Matrix(flat, N, M)

    @property
    def vector(self):
        '''Vector/displacement part of the transformation'''

        N, M = self.shape
        data = self.flat
        return asvector([data[M + i + i * M] for i in range(N)])

    @property
    def n_rows(self):
        '''Number of rows for the matrix part'''

        return self.shape[0]

    @property
    def n_cols(self):
        '''Number of columns for the matrix part'''

        return self.shape[1]

    #
    # Data representation
    #
    def as_matrix(self):
        '''Return a minimum matricial representation of the affine
        transformation'''

        return self.matrix.append_col(self.vector)

    def as_augmented_matrix(self):
        '''Return the augmented matrix representation of the affine transform.
        The augmented matrix operates linearly in one dimension greater than
        the original transform. The trick is to append 1 to the last component
        of a vector in order to mimick the same results of the affine
        transformation using strictly linear operations.'''

        # FIXME: what happens in non uniform dimensions?
        bottom_row = [0] * (self.shape[1] + 1)
        bottom_row[-1] = 1
        return self.matrix.append_col(self.vector).append_row(bottom_row)

    #
    # Magic methods
    #
    def __repr__(self):
        matrix = str(self.matrix).splitlines()
        vector = str(Matrix([[x] for x in self.vector])).splitlines()
        lines = ((x + y) for (x, y) in zip(matrix, vector))
        lines = (line.replace('||', ' : ') for line in lines)
        return '\n'.join(lines)

    def __eq__(self, other):
        return self.matrix == other.matrix and self.vector == other.vector

    # Behaves somewhat as a tuple of (matrix, vector). Useful for extracting
    # both of these components in the same line.
    #     M, b = affine
    def __iter__(self):
        yield self.matrix
        yield self.vector

    def __getitem__(self, idx):
        if idx in (0, -2):
            return self.matrix
        elif idx in (1, -1):
            return self.vector
        else:
            raise IndexError(idx)

    def __len__(self):
        return 2

    # Call delegates to multiplication with the proper object
    def __call__(self, *args):
        if len(args) > 1:
            return self * args
        else:
            return self * args[0]

    #
    # Arithmetic operations
    #
    def __mul__(self, other):
        if isinstance(other, (Vec, tuple, list)):
            return self.matrix * other + self.vector
        else:
            return Affine(self.matrix * other, self.vector * other)

    def __rmul__(self, other):
        return Affine(other * self.matrix, other * self.vector)

    def __sub__(self, other):
        return Affine(self.matrix, self.vector - other)

    def __add__(self, other):
        return Affine(self.matrix, self.vector + other)


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
