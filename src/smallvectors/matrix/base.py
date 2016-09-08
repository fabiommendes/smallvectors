from generic import overload, convert
from generic.op import mul

from smallvectors import Vec, asvector
from smallvectors.core import SmallVectorsBase, AddElementWise
from smallvectors.core.mutability import Mutable, Immutable
from smallvectors.matrix.mat2x2 import Mat2x2Mixin
from smallvectors.matrix.mat3x3 import Mat3x3Mixin
from smallvectors.matrix.square import SquareMixin
from smallvectors.utils import flatten, dtype as _dtype
from smallvectors.vector.vec import VecAny

__all__ = [
    # Types
    'Mat', 'mMat', 'MatAny',

    # Functions
    'asmatrix', 'asamatrix', 'asmmatrix', 'identity', 'midentity',
]
number = (float, int)


class MatAny(SmallVectorsBase, AddElementWise):
    """
    Base class for mutable and immutable matrix types
    """

    __slots__ = ()
    __parameters__ = (int, int, type)

    @classmethod
    def __preparenamespace__(cls, params):
        ns = SmallVectorsBase.__preparenamespace__(params)
        try:
            m, n, dtype = params
        except ValueError:
            pass
        else:
            if (m, n) not in [(2, 2), (3, 3)]:
                ns['__slots__'] = 'flat'
        return ns

    @classmethod
    def __preparebases__(cls, params):
        N, M, dtype = params
        if isinstance(N, int) and isinstance(M, int):
            if N == M == 2:
                return Mat2x2Mixin, cls
            elif N == M == 3:
                return Mat3x3Mixin, cls
            elif M == N:
                return SquareMixin, cls
        return (cls,)

    @staticmethod
    def __finalizetype__(cls):
        SmallVectorsBase.__finalizetype__(cls)
        if cls.shape:
            cls.nrows, cls.ncols = cls.shape
        else:
            cls.nrows = cls.ncols = None

    #
    # Constructors
    #
    @classmethod
    def __abstract_new__(cls, *args, dtype=None):
        flat, nrows, ncols = flatten(args, 2)
        return cls[nrows, ncols].from_flat(flat)

    def __init__(self, *args):
        flat, N, M = flatten(args, 2)
        self.flat = self.__flat__(flat, copy=False)
        if N != self.nrows or M != self.ncols:
            raise ValueError('data has an invalid shape: %s' % repr((N, M)))

    @classmethod
    def from_dict(cls, D):
        """
        Build a matrix from a dictionary from indexes to values.
        """

        N, M = cls.shape
        data = [0] * (N * M)
        for (i, j), value in D.items():
            if i >= cls.nrows or j >= cls.nrows:
                fmt = (N, M, (i, j))
                raise IndexError('invalid index for %sx%s matrix: %r' % fmt)
            k = i * N + j
            data[k] = value

        return cls.from_flat(data)

    @classmethod
    def from_rows(cls, rows, dtype=None):
        """
        Build matrix from a sequence of row vectors.
        """

        data, M, N = flatten(rows, 2)
        if (M, N) != cls.shape:
            if cls.size is None:
                return cls[M, N, cls.dtype].from_flat(data, dtype)
            msg = ('data shape %s is not consistent with matrix type %s' %
                   ((M, N), cls.shape))
            raise ValueError(msg)
        return cls.from_flat(data, dtype=dtype)

    @classmethod
    def from_cols(cls, cols, dtype=None):
        """
        Build matrix from a sequence of column Vecs.
        """

        dataT, N, M = flatten(cols, 2)
        data = dataT[:]
        for i in range(M):
            data[i * N:i * N + N] = dataT[i::M]

        if (M, N) != cls.shape:
            if cls.size is None:
                return cls[M, N, cls.dtype].from_flat(data, dtype)
            msg = ('data shape %s is not consistent with matrix type %s' %
                   ((M, N), cls.shape))
            raise ValueError(msg)
        return cls.from_flat(data, dtype=dtype)

    # Attributes
    @property
    def T(self):
        """
        Matrix transpose.
        """
        return self.transpose()

    def as_dict(self):
        """
        Return matrix data as a map from indexes to (non-null)
        values.
        """

        D = {}
        M, N = self.shape
        for i in range(M):
            for j in range(N):
                value = self[i, j]
                if value == 0:
                    D[i, j] = value
        return D

    # Iterators
    def items(self):
        """
        Iterator over ((i, j), value) pairs.

        Example:
            >>> matrix = Mat([1, 2], [3, 4])
            >>> for ((i, j), k) in matrix.items():
            ...     print('M[%s, %s] = %s' % (i, j, k))
            M[0, 0] = 1
            M[0, 1] = 2
            M[1, 0] = 3
            M[1, 1] = 4
        """

        N = self.nrows
        for k, value in enumerate(self.flat):
            i = k // N
            j = k % N
            yield (i, j), value

    def cols(self):
        """
        Iterator over columns of matrix

        See also:
            :meth:`rows`: iterate over the rows of a matrix.

        Example:
            >>> M = Mat([1, 2],
            ...         [3, 4])

            >>> list(M.cols())
            [Vec(1, 3), Vec(2, 4)]

            >>> list(M.rows())
            [Vec(1, 2), Vec(3, 4)]
        """

        M = self.ncols
        data = self.flat
        for i in range(M):
            yield asvector(data[i::M])

    def rows(self):
        """
        Iterator over row vectors.
        """

        return iter(self)

    def col(self, i):
        """
        Return the i-th column Vec.
        """

        return asvector(self.flat[i::self.ncols])

    def row(self, i):
        """
        Return the i-th row Vec.

        Same as matrix[i], but exists for symmetry with the matrix.col(i)
        method.
        """

        M = self.ncols
        start = i * M
        return asvector(self.flat[start:start + M])

    def copy(self, D=None, **kwds):
        """
        Return a copy of matrix possibly changing some terms.

        Can set terms in the matrix by either passing a mapping from (i, j)
        indexes to values or by passing keyword argments of the form
        ``Aij=value``. The keyword form only accept single digit indexes

        Example
        -------

        Copy can do a simple copy of matrix or can do a copy with some
        overrides

        >>> matrix = Mat([1, 2], [3, 4])

        Let us override using a dict

        >>> matrix.copy({(0, 0): 42, (0, 1): 3})
        Mat([42, 3], [3, 4])

        A similar operation can also be done by setting the corresponding
        keyword arguments. Remember that indexes start at zero as usual.

        >>> matrix.copy(A00=42, A01=3)
        Mat([42, 3], [3, 4])
        """

        N, matrix = self.shape

        # Populate the dictionary D with values from keywords
        if kwds:
            D = dict(D or {})
            for k, v in kwds.items():
                try:
                    name, i, j = k
                    if name == 'a':
                        raise ValueError
                    i, j = int(i), int(j)
                    if i >= N or j >= N:
                        raise ValueError

                except (ValueError, IndexError):
                    raise TypeError('invalid keyword argument: %s' % k)

                else:
                    D[i, j] = v

        if D is None or not D:
            data = self.flat
        else:
            data = list(self.flat)
            for (i, j), value in D.items():
                if j >= matrix or i >= N:
                    msg = 'invalid index for matrix: %s' % str((i, j))
                    raise IndexError(msg)
                k = i * N + j
                data[k] = value

        return self.from_flat(data)

    def transpose(self):
        """
        Return the transposed matrix

        Example
        -------

        Transpose just mirrors items around the diagonal.

        >>> matrix = Mat([1, 2], [3, 4])
        >>> print(matrix)
        |1  2|
        |3  4|

        >>> print(matrix.transpose())
        |1  3|
        |2  4|

        ``matrix.T`` is an alias to the transpose() method

        >>> matrix.T == matrix.transpose()
        True
        """

        N, M = self.shape
        if N == M:
            return self.from_cols(self.rows())
        else:
            return self.__origin__[M, N, self.dtype].from_cols(self.rows())

    # Shape transformations
    def __raise_badshape(self, data, s1):
        fmt = data, s1, self.nrows, self.nrows
        raise ValueError('incompatible %s size: %s with %s x %s matrix' % fmt)

    def append_row(self, data, index=None):
        """
        Return a new matrix with extra rows inserted in the given index.
        If no index is given, insert rows at the end.

        If the argument ``data`` is a matrix, multiple rows are inserted.
        """

        N, M = self.shape
        flat = list(self.flat)

        if isinstance(data, Mat):
            if data.ncols != M:
                self.__raise_badshape('row', data.ncols)
            extra = list(data.flat)
            N += data.nrows
        else:
            extra = list(data)
            if len(data) != M:
                self.__raise_badshape('row', len(data))
            N += 1

        if index is None:
            flat.extend(extra)
        else:
            index = M * index
            flat = flat[:index] + extra + flat[index:]

        # FIXME: infer correct dtype
        T = self.__origin__[N, M, self.dtype]
        return T.from_flat(flat, copy=False)

    def append_col(self, data, index=None):
        """
        Similar to M.append_row(), but works with columns.
        """

        N, M = self.shape
        cols = list(self.cols())
        if isinstance(data, Mat):
            if data.nrows != N:
                self.__raise_badshape('column', data.ncols)
            if index is None:
                cols.extend(data.cols())
            else:
                cols = cols[:index] + list(data.cols) + cols[index:]
            M += len(data)

        else:
            if len(data) != N:
                self.__raise_badshape('row', len(data))
            if index is None:
                cols.append(list(data))
            else:
                cols.insert(index, list(data))
            M += 1

        # FIXME: infer correct dtype
        T = self.__origin__[N, M, self.dtype]
        return T.from_cols(cols)

    def drop_col(self, idx=-1):
        """
        Drops column in the given index.

        Return a pair (matrix, col) with the new matrix with the extra column
        removed
        """

        M, N = self.shape
        data = list(self.cols())
        col = data.pop(idx)
        T = self.__origin__[M, N - 1, self.dtype]
        return T.from_cols(data), col

    def drop_row(self, idx=-1):
        """
        Return a pair (matrix, row) with the new matrix with the extra row
        removed.
        """

        M, N = self.shape
        data = list(self.rows())
        row = data.pop(idx)
        T = self.__origin__[M - 1, N, self.dtype]
        return T.from_rows(data), row

    def select_cols(self, *indexes):
        """
        Return a new matrix with the columns with given indexes.
        """

        L = list(self.cols())
        data = [L[i] for i in indexes]
        T = self.__origin__[self.nrows, len(data), self.dtype]
        return T.from_cols(data)

    def select_rows(self, *indexes):
        """
        Return a new matrix with the rows corresponding to the given
        sequence of indexes.
        """

        L = list(self.rows())
        data = [L[i] for i in indexes]
        T = self.__origin__[len(data), self.ncols, self.dtype]
        return T.from_rows(data)

    # Magic methods
    def __repr__(self):
        data = ', '.join([repr(list(x)) for x in self])
        return '%s(%s)' % (self.__origin__.__name__, data)

    def __str__(self):
        def fmt(x):
            if isinstance(x, float):
                return ('%.3f' % x).rstrip('0').rstrip('.')
            else:
                return repr(x)

        N, M = self.shape
        data = list(map(fmt, self.flat))
        cols = [data[i::M] for i in range(M)]
        sizes = [max(map(len, col)) for col in cols]
        cols = [[x.rjust(k) for x in col] for (col, k) in zip(cols, sizes)]
        lines = ['|%s|' % '  '.join(line) for line in zip(*cols)]
        return '\n'.join(lines)

    def __len__(self):
        return self.nrows

    def __iter__(self):
        N, M = self.shape
        data = self.flat
        start = 0
        for _ in range(N):
            yield asvector(data[start:start + M])
            start += M

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, j = idx
            return self.flat[i * self.ncols + j]

        elif isinstance(idx, int):
            return self.row(idx)

    # Arithmetic operations
    def __mul__(self, other):
        if isinstance(other, (Vec, tuple, list)):
            other = asvector(other)
            return asvector([u.dot(other) for u in self.rows()])

        elif isinstance(other, number):
            return self.from_flat([x * other for x in self.flat], copy=False)

        elif isinstance(other, Mat):
            cols = list(other.cols())
            rows = list(self.rows())
            data = sum([[u.dot(v) for u in cols] for v in rows], [])
            cls = self.__origin__[len(rows), len(cols), _dtype(data)]
            return cls.from_flat(data, copy=False)

        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, number):
            return self.from_flat([x * other for x in self.flat], copy=False)
        else:
            other = asvector(other)
            return asvector([u.dot(other) for u in self.cols()])

    def __div__(self, other):
        if isinstance(other, number):
            return self.from_flat(x / other for x in self.flat)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, number):
            return self.from_flat(x / other for x in self.flat)
        else:
            return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, number):
            return self.from_flat(x // other for x in self.flat)
        else:
            return NotImplemented

    def _zip_other(self, other):
        """Auxiliary function that zip the flat iterator of both operands if
        the shapes are valid"""

        if not isinstance(other, Mat):
            raise NotImplementedError

        if (self.ncols != other.ncols) or (self.nrows != other.nrows):
            n, m = self.shape
            a, b, = other.shape
            msg = 'not aligned: [%s x %s] to [%s x %s]' % (n, m, a, b)
            raise ValueError(msg)

        return zip(self.flat, other.flat)

    def __add__(self, other):
        return self.from_flat(x + y for (x, y) in self._zip_other(other))

    def __radd__(self, other):
        return self.from_flat(y + x for (x, y) in self._zip_other(other))

    def __sub__(self, other):
        return self.from_flat(x - y for (x, y) in self._zip_other(other))

    def __rsub__(self, other):
        return self.from_flat(y - x for (x, y) in self._zip_other(other))

    def __neg__(self):
        return self.from_flat(-x for x in self.flat)

    def __nonzero__(self):
        return True


class Mat(MatAny, Immutable):
    """
    A immutable matrix type.
    """

    __slots__ = ()


class mMat(MatAny, Mutable):
    """
    A mutable Mat.
    """

    __slots__ = ()

    def iswap_cols(self, i, j):
        """
        Swap columns i and j *INPLACE*"""

        if i != j:
            M = self.nrows
            self.flat[i::M], self.flat[j::M] = self.flat[j::M], self.flat[i::M]

    def iswap_rows(self, i, j):
        """Swap rows i and j *INPLACE*"""

        if i != j:
            self[i], self[j] = self[j], self[i]

    def isum_row(self, i, vec):
        """
        Adds the contents of a vector into the i-th row *INPLACE*
        """

        N, M, dtype = self.__parameters__
        if i < 0:
            i = M - i
        data = self.flat
        for j in range(M):
            data[i * M + j] = convert(data[i * M + j] + vec[j], dtype)

    def isum_col(self, i, vec):
        """
        Adds the contents of a vector into the i-th column *INPLACE*.
        """

        N, M, dtype = self.__parameters__
        if i < 0:
            i = N - i
        data = self.flat
        for j in range(N):
            data[j * M + i] = convert(data[j * M + i] + vec[j], dtype)

    def imul_row(self, i, value):
        """
        Multiply the i-th row by the given value *INPLACE*.
        """

        N, M, dtype = self.__parameters__
        if i < 0:
            i = M - i
        data = self.flat
        for j in range(M):
            data[i * N + j] = convert(data[i * N + j] * value, dtype)

    def imul_col(self, i, value):
        """
        Multiply the i-th column by the given value *INPLACE*.
        """

        N, M, dtype = self.__parameters__
        if i < 0:
            i = N - i
        data = self.flat
        for j in range(N):
            data[j * N + i] = convert(data[j * N + i] * value, dtype)

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            i, j = idx
            self.flat[i * self.ncols + j] = convert(value, self.dtype)
        elif isinstance(idx, int):
            M = self.ncols
            start = idx * M
            dtype = self.dtype
            self.flat[start:start + M] = [convert(x, dtype) for x in value]


# Overloads and promotions
@overload(mul, (Mat, Vec))
def mul_matrix_Vec(M, v):
    return NotImplemented


@overload(mul, (Vec, Mat))
def mul_Vec_matrix(v, M):
    return NotImplemented


# Matrices conversions
def asmatrix(m):
    """
    Return object as an immutable matrix.
    """

    if isinstance(m, Mat):
        return m
    else:
        return Mat(*m)


def asmmatrix(m):
    """
    Return object as a mutable matrix.
    """

    if isinstance(m, mMat):
        return m
    else:
        return mMat(*m)


def asamatrix(m):
    """
    Return object as a matrix (mutable or immutable).
    """

    if isinstance(m, MatAny):
        return m
    else:
        return Mat(*m)


def identity(N, dtype=float):
    """
    Return an identity matrix of size N by N.
    """

    return Mat[N, N, dtype].from_diag([1] * N)


def midentity(N, dtype=float):
    """
    Return a mutable identity matrix of size N by N.
    """

    return mMat[N, N, dtype].from_diag([1] * N)


SquareMixin._matrix = Mat
SquareMixin._mmatrix = mMat
SquareMixin._identity = identity
SquareMixin._midentity = midentity
SquareMixin._rotmatrix = Mat
VecAny._rotmatrix = Mat
