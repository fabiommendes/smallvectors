# -*- coding: utf8 -*-
from __future__ import division
from generic import overload, promote_type
from generic.operator import mul
from smallvectors.core import BaseAbstractType, shape, dtype, get_common_base, Immutable, Mutable
from smallvectors import asvector, Vec

__all__ = ['Mat']
number = (float, int)
dtype_ = dtype


class AnyMat(BaseAbstractType):

    '''Generic matrix interface

    Example
    -------

    Criamos uma matriz a partir de uma lista de listas

    >>> matrix = Mat([1, 2],
    ...              [3, 4])

    Podemos também utilizar classes especializadas, como por exemplo
    a `RotMat`, que cria uma matriz de rotação

    >>> R = Mat([-1, 0],
    ...         [0, -1]); R
    |-1   0|
    | 0  -1|

    Os objetos da classe Mat implementam as operações algébricas básicas

    >>> matrix + 2 * R
    |-1  2|
    | 3  2|

    As multiplicações são como definidas em ágebra linear

    >>> matrix * matrix
    | 7  10|
    |15  22|

    Onde multiplicação por vetores também é aceita

    >>> v = Vec(2, 3)
    >>> matrix * v
    Vec[2, int](8, 18)

    Note que não existem classes especializadas para vetores linha ou coluna.
    Deste modo, sempre assumimos o formato que permite realizar a
    multiplicação.

    >>> v * matrix   # agora v é tratado como um vetor linha
    Vec[2, int](11, 16)

    Além disto, temos operações como cálculo da inversa, autovalores,
    determinante, etc

    >>> matrix.inv() * matrix
    |1  0|
    |0  1|

    #>>> (matrix * matrix.inv()).eigval()
    #(1.0, 1.0)
    '''
    __slots__ = ()

    @classmethod
    def __abstract_new__(cls, *args, **_3to2kwargs):
        if 'dtype' in _3to2kwargs: dtype = _3to2kwargs['dtype']; del _3to2kwargs['dtype']
        else: dtype = None
        shape_ = shape(args)
        try:
            nrows, ncols = shape_
        except ValueError:
            raise ValueError('invalid input shape: %s' % repr(shape_))
        if dtype is None:
            dtype = get_common_base(*[dtype_(x) for x in args])
        return cls[nrows, ncols, dtype](*args)

    def __init__(self, *args):
        self.flat = self._flat(sum(map(list, args), []), copy=False)

    ndims = 2

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    @classmethod
    def identity(cls, N):
        '''Return an identity matrix of size N by N'''

        return cls.from_diag([1] * N)

    @classmethod
    def from_dict(cls, D):
        '''Build a matrix from a dictionary from indexes to values'''

        N = max(i for (i, _j) in D)
        matrix = max(j for (_i, j) in D)
        data = [0.0] * (N * matrix)
        for (i, j), value in D.items():
            k = i * N + j
            data[k] = value

        return cls.from_flat(data, N, matrix)

    @classmethod
    def from_lists(cls, L):
        '''Build matrix from a sequence of lists.'''

        return cls.from_rows([asvector(Li) for Li in L])

    @classmethod
    def from_rows(cls, rows):
        '''Build matrix from a sequence of row Vecs'''

        N = len(rows)
        M = len(rows[0])
        data = []
        for row in rows:
            if len(row) != M:
                raise ValueError('Vecs must be of same length')
            data.extend(row)
        return cls.__root__[N, M, dtype(data)].from_flat(data)

    @classmethod
    def from_cols(cls, cols):
        '''Build matrix from a sequence of column Vecs'''

        return cls.from_rows(cols).T

    #
    # Attributes
    #
    @property
    def shape(self):
        return self.nrows, self.ncols

    @property
    def size(self):
        return self.nrows * self.ncols

    @property
    def T(self):
        return self.transpose()

    is_mutable = False

    #
    # Return row Vecs or column Vecs
    #
    def as_lists(self):
        '''Return matrix data as a list of lists'''

        return [list(row) for row in self]

    def as_dict(self):
        '''Return matrix data as a mapping from indexes to (non-null)
        values.'''

    def items(self):
        '''Iterator over ((i, j), value) pairs.

        Example
        -------

        >>> matrix = Mat([1, 2], [3, 4])
        >>> for ((i, j), k) in matrix.items():
        ...     print('M[%s, %s] = %s' % (i, j, k))
        M[0, 0] = 1
        M[0, 1] = 2
        M[1, 0] = 3
        M[1, 1] = 4
        '''

        N = self.nrows
        for k, value in enumerate(self.flat):
            i = k // N
            j = k % N
            yield (i, j), value

    def col(self, i):
        '''Return the i-th column Vec'''

        matrix = self.ncols
        return asvector(self.flat[i::matrix])

    def row(self, i):
        '''Return the i-th row Vec.

        Same as matrix[i], but exists for symmetry with the matrix.col(i)
        method.'''

        matrix = self.ncols
        start = i * matrix
        return asvector(self.flat[start:start + matrix])

    def colvecs(self):
        '''Return list of all column vectors

        See also
        --------

        :meth:`rowvecs`: return the row vectors of a matrix.

        Example
        -------

        >>> M = Mat([1, 2], [3, 4])

        >>> M.colvecs()
        [Vec[2, int](1, 3), Vec[2, int](2, 4)]

        >>> M.rowvecs()
        [Vec[2, int](1, 2), Vec[2, int](3, 4)]

        '''

        matrix = self.ncols
        data = self.flat
        return [asvector(data[i::matrix]) for i in range(matrix)]

    def rowvecs(self):
        '''Return list of all row Vecs. Same as list(matrix)'''

        return list(self)

    def copy(self, D=None, **kwds):
        '''Return a copy of matrix possibly changing some terms.

        Can set terms in the matrix by either passing a mapping from (i, j)
        indexes to values or by passing keyword argments of the form
        ``Aij=value``. The keyword form only accept single digit indexes

        Example
        -------

        Copy can do a simple copy of matrix or can do a copy with some
        overrides

        >>> matrix = Mat([1, 2], [3, 4])

        Let us override using a dict

        >>> matrix.copy({(0, 0): 42})
        |42  2|
        | 3  4|

        A similar operation can also be done by setting the corresponding
        keyword arguments. Remember that indexes start at zero as usual.

        >>> matrix.copy(A00=42, A01=3)
        |42  3|
        | 3  4|

        '''

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

    def mutable(self, D=None, **kwds):
        '''Like copy(), but always return a mutable object'''

        cp = self.copy(D, **kwds)
        if self.is_mutable:
            return cp
        else:
            cls = mMat[self.nrows, self.ncols, self.dtype]
            return cls.from_flat(list(cp.flat))

    def immutable(self, D=None, **kwds):
        '''Like copy(), but always return an immutable object'''

        cp = self.copy(D, **kwds)
        if not self.is_mutable:
            return cp
        else:
            return cp._immutable_t.from_flat(cp.flat, *cp.shape)

    def transpose(self):
        '''Return the transposed matrix

        Example
        -------

        Transpose just mirrors items around the diagonal.

        >>> matrix = Mat([1, 2], [3, 4]); matrix
        |1  2|
        |3  4|

        >>> matrix.transpose()
        |1  3|
        |2  4|

        ``matrix.T`` is an alias to the transpose() method

        >>> matrix.T == matrix.transpose()
        True
        '''

        return self.from_rows(self.colvecs())

    def append_row(self, row):
        '''Return a new matrix with and extra row appended to the end.

        If the argument ``row`` is a matrix, multiple rows are inserted.'''

        N, matrix = self.shape
        data = list(self.flat)
        data.extend(row)
        return self.from_flat(data, N + 1, matrix)

    def append_col(self, col, idx=-1):
        '''Return a new matrix with and extra column appended to the end.

        If the argument ``col`` is a matrix, multiple columns are inserted.'''

        if isinstance(col, Mat):
            matrix = col
            out = self.as_lists()
            for row, L in zip(matrix, out):
                L.extend(row)
            return self.from_lists(out)
        else:
            out = self.as_lists()
            for x, L in zip(col, out):
                L.append(x)
            return self.from_lists(out)

    def drop_col(self, idx=None):
        '''Return a pair (matrix, col) with the new matrix with the extra column
        removed'''

        transpose, col = self.T.drop_row(idx)
        return transpose.T, col

    def drop_row(self, idx=None):
        '''Return a pair (matrix, row) with the new matrix with the extra row
        removed'''

        if idx is None:
            idx = -1
        data = self.as_lists()
        row = data.pop(idx)
        return self.from_lists(data), asvector(row)

    def select_cols(self, cols):
        '''Return a new matrix with the columns corresponding to the given
        sequence of indexes'''

        cols = tuple(cols)
        data = [[L[i] for i in cols] for L in self.as_lists()]
        return self.from_lists(data)

    #
    # Magic methods
    #
    def _fmt_number(self, x):
        # TODO: fix this to all matrices
        return ('%.3f' % x).rstrip('0').rstrip('.')

    def __repr__(self):
        N, matrix = self.shape
        fmt = self._fmt_number
        rows = [[fmt(x) for x in row] for row in self.rowvecs()]
        sizes = sum([[len(x) for x in row] for row in rows], [])
        sizes_T = [sizes[i::N] for i in range(matrix)]
        sizes_max = [max(col) for col in sizes_T]
        rows = [[x.rjust(n) for n, x in zip(sizes_max, row)] for row in rows]
        rows = ['|%s|' % ('  '.join(row)) for row in rows]
        return '\n'.join(rows)

    def __str__(self):
        return repr(self)

    def __len__(self):
        return self.nrows

    def __iter__(self):
        N, matrix = self.shape
        data = self.flat
        start = 0
        for _ in range(N):
            yield asvector(data[start:start + matrix])
            start += matrix

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, j = idx
            return self.flat[i * self.ncols + j]

        elif isinstance(idx, int):
            return self.row(idx)

    def __eq__(self, other):
        if self.ncols != other.ncols or self.nrows != other.nrows:
            return False
        else:
            return all(x == y for (x, y) in zip(self.flat, other.flat))

    #
    # Arithmetic operations
    #
    def __mul__(self, other):
        if isinstance(other, (Vec, tuple, list)):
            other = asvector(other)
            return asvector([u.dot(other) for u in self.rowvecs()])

        elif isinstance(other, number):
            return self.from_flat([x * other for x in self.flat], copy=False)

        elif isinstance(other, Mat):
            cols = other.colvecs()
            rows = self.rowvecs()
            data = sum([[u.dot(v) for u in cols] for v in rows], [])
            cls = self.__root__[len(rows), len(cols), dtype(data)]
            return cls.from_flat(data, copy=False)

        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, number):
            return self.from_flat([x * other for x in self.flat], copy=False)
        else:
            other = asvector(other)
            return asvector([u.dot(other) for u in self.colvecs()])

    def __div__(self, other):
        if isinstance(other, number):
            return self._from_flat(x / other for x in self.flat)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, number):
            return self._from_flat(x / other for x in self.flat)
        else:
            return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, number):
            return self._from_flat(x // other for x in self.flat)
        else:
            return NotImplemented

    def _zip_other(self, other):
        '''Auxiliary function that zip the flat iterator of both operands if
        the shapes are valid'''

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

    #
    # Specialized methods for square matrices
    #
    @classmethod
    def from_diag(cls, diag):
        '''Create diagonal matrix from diagonal terms'''

        N = len(diag)
        data = [0] * (N * N)
        for i in range(N):
            data[N * i + i] = diag[i]
        return cls.__root__[N, N, dtype(data)].from_flat(data, copy=False)

    def det(self):
        '''Return the determinant of the matrix'''

        raise NotImplementedError

    def trace(self):
        '''Computes the trace (i.e., sum of all elements in the diagonal)'''

        N = self.nrows
        return sum(i * N + i for i in range(N))

    def diag(self):
        '''Return a Vec with the diagonal elements of the matrix'''

        N = self.nrows
        data = self.flat
        return asvector([data[i * N + i] for i in range(N)])

    def with_diag(self, diag):
        '''Return a copy of the matrix set with the given diagonal terms'''

        N = self.nrows
        if len(diag) != N:
            raise ValueError('wrong size for diagonal')

        # Write diagonal in a copy of flat data
        data = list(self.flat)
        for i, x in zip(range(N), diag):
            data[i * N + i] = x

        return self._from_flat(data)

    def nondiag(self):
        '''Return a matrix stripped from the diagonal'''

        N = self.nrows
        data = list(self.flat)
        for i in range(N):
            data[i * N + i] *= 0
        return self._from_flat(data)

    def eig(self):
        '''Retorna uma tupla com a lista de autovalores e a matriz dos
        autovetores

        Example
        -------

        Criamos uma matriz e aplicamos eig()

        #>>> matrix = Mat([[1,2], [3,4]])
        #>>> vals, vecs = matrix.eig()

        Agora extraimos os auto-vetores coluna

        #>>> v1, v2 = vecs.colvecs()

        Finalmente, multiplicamos matrix por v1: o resultado deve ser igual que
        multiplicar o autovetor pelo autovalor correspondente

        #>>> matrix * v1, vals[0] * v1
        #(Vec2(2.2, 4.9), Vec2(2.2, 4.9))
        '''

        raise NotImplementedError

    def eigval(self):
        '''Retorna uma tupla com os autovalores da matriz'''

        raise NotImplementedError

    def eigasvector(self, transpose=False):
        '''Retorna uma lista com os autovetores normalizados da matriz.

        A ordem dos autovetores corresponde àquela retornada pelo método
        `matrix.eigval()`'''

        raise NotImplementedError

    def inv(self):
        '''Returns the inverse matrix'''

        # Simple and naive matrix inversion
        # Creates extended matrix
        N, M = self.shape
        dtype = promote_type(float, self.dtype)
        matrix = mMat[N, M, dtype].from_flat(self.flat)
        matrix = matrix.append_col(self.identity(N))

        # Make left hand side upper triangular
        for i in range(0, N):
            # Search for maximum value in the truncated column and put it
            # in pivoting position
            trunc_col = list(matrix.col(i))[i:]
            _, idx = max([(abs(c), i) for (i, c) in enumerate(trunc_col)])
            matrix.iswap_rows(i, idx + i)

            # Find linear combinations that make all rows below the current one
            # become equal to zero in the current column
            Z = matrix[i, i]
            for k in range(i + 1, N):
                matrix.irow_add(k, i, -matrix[k, i] / Z)

            # Make the left hand side diagonal
            Z = matrix[i, i]
            for k in range(0, i):
                matrix.irow_add(k, i, -matrix[k, i] / Z)

        # Normalize by the diagonal
        for i in range(N):
            matrix.irow_mul(i, 1 / matrix[i, i])

        out = matrix.select_cols(range(M, 2 * M))
        return Mat[N, N, dtype].from_flat(out.flat)

    def solve(self, b, method='gauss', **kwds):
        '''Solve the linear system ``matrix * x  = b`` for x.'''

        method = getattr(self, 'solve_%s' % method)
        return method(b, **kwds)

    def solve_jacobi(self, b, tol=1e-3, x0=None, maxiter=None):
        '''Solve a linear using Gauss-Jacobi method'''

        b = asvector(b)
        x = b * 0
        D = self.from_diag([1.0 / x for x in self.diag()])
        R = self.nondiag()

        for _ in range(maxiter or int(1e6)):
            x, old = D * (b - R * x), x
            if (x - old).norm() < tol:
                break
        return x

    def solve_gauss(self, b):
        '''Solve system by simple Gaussian elimination'''

        # Creates extended matrix
        N = self.nrows
        matrix = self.mutable().append_col(b)
        for i in range(0, N):
            # Search for maximum value in the truncated column and put it
            # in pivoting position
            trunc_col = list(matrix.col(i))[i:]
            _, idx = max([(abs(c), i) for (i, c) in enumerate(trunc_col)])
            matrix.iswap_rows(i, idx + i)

            # Find linear combinations that make all rows below the current one
            # become equal to zero in the current column
            Z = matrix[i, i]
            for k in range(i + 1, N):
                matrix.irow_add(k, i, -matrix[k, i] / Z)

        # Solve equation Ax=b for an upper triangular matrix
        A, b = matrix.drop_col()
        x = [0] * N
        for i in range(N - 1, -1, -1):
            x[i] = (b[i] - sum(A * x)) / A[i, i]
        return x


class Mat(AnyMat, Immutable):
    pass


class mMat(Mutable, AnyMat):
    is_mutable = True
    _immutable_t = Mat

    __slots__ = ()

    def iswap_cols(self, i, j):
        '''Swap columns i and j *inplace*'''

        raise NotImplementedError

    def iswap_rows(self, i, j):
        '''Swap rows i and j *inplace*'''

        if i == j:
            return

        matrix = self.ncols
        data = self.flat
        start_i = i * matrix
        start_j = j * matrix
        for r in range(matrix):
            ki = start_i + r
            kj = start_j + r
            data[ki], data[kj] = data[kj], data[ki]

    def irow_add(self, i, j, alpha=1):
        '''Adds alpha times row j to row i *inplace*'''

        matrix = self.ncols
        data = self.flat
        start_i = i * matrix
        start_j = j * matrix
        for r in range(matrix):
            ki = start_i + r
            kj = start_j + r
            data[ki] += alpha * data[kj]

    def irow_mul(self, i, value):
        '''Multiply row by the given value *inplace*'''

        matrix = self.ncols
        data = self.flat
        for j in range(matrix):
            data[i * matrix + j] *= value

    def icol_add(self, i, j, alpha=1):
        '''Adds alpha times column j to column i *inplace*'''

        raise NotImplementedError

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            i, j = idx
            k = self.nrows * i + j
            self.flat[k] = value
        else:
            data = self.flat
            for k, x in zip(range(idx, idx + self.ncols), value):
                data[k] = x

Mat._mutable_t = mMat


@overload(mul, (Mat, Vec))
def mul_matrix_Vec(M, v):
    return NotImplemented


@overload(mul, (Vec, Mat))
def mul_Vec_matrix(v, M):
    return NotImplemented

if __name__ == '__main__':
    u = Vec(1, 2)
    matrix = Mat([1, 2], [3, 4])
    print(matrix)
    print(matrix * matrix)
    print(matrix.inv())
    print(matrix * matrix.inv(), '\n')

    M = Mat([0, 1])
    print(M, '\n')
    print(M.T, '\n')
    M = Mat([0], [1])
    print(M)

    # print(matrix)
    v = matrix.solve(u)
    print(u)
    print(matrix * v)
    print(u)

    import doctest
    doctest.testmod()
