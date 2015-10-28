'''
========
Matrices
========

Generic matrix interface

Example
-------

Criamos uma matriz a partir de uma lista de listas

>>> matrix = Mat([1, 2],
...              [3, 4])

Podemos também utilizar classes especializadas, como por exemplo
a `RotMat`, que cria uma matriz de rotação

>>> R = Mat([-1, 0],
...         [0, -1])
>>> print(R)
|-1   0|
| 0  -1|

Os objetos da classe Mat implementam as operações algébricas básicas

>>> print(matrix + 2 * R)
|-1  2|
| 3  2|

As multiplicações são como definidas em ágebra linear

>>> print(matrix * matrix)
| 7  10|
|15  22|

Onde multiplicação por vetores também é aceita

>>> v = Vec(2, 3)
>>> matrix * v
Vec(8, 18)

Note que não existem classes especializadas para vetores linha ou coluna.
Deste modo, sempre assumimos o formato que permite realizar a
multiplicação.

>>> v * matrix   # agora v é tratado como um vetor linha
Vec(11, 16)

Além disto, temos operações como cálculo da inversa, autovalores,
determinante, etc

>>> print(matrix.inv() * matrix)
|1  0|
|0  1|

>>> (matrix * matrix.inv()).eigenvalues()
[1.0, 1.0]
    '''


from generic import overload, promote_type, convert
from generic.operator import mul
from smallvectors.core import SmallVectorsBase, Immutable, Mutable, AddElementWise
from smallvectors.core import shape, dtype as _dtype, get_common_base, FlatView, Any
from smallvectors import Vec, asvector

__all__ = ['Mat', 'mMat', 'MatAny']
number = (float, int)


class MatAny(SmallVectorsBase, AddElementWise):
    '''
    Base class for mutable and immutable matrix types
    '''
    
    __slots__ = ()
    __parameters__ = (int, int, type)

    @classmethod
    def __preparenamespace__(cls, params):
        ns = SmallVectorsBase.__preparenamespace__(params)
        return ns

    @classmethod
    def __preparebases__(cls, params):
        N, M, dtype = params
        if (isinstance(N, int) and isinstance(M, int)):
            if N == M == 2:
                return (Mat2x2, cls)
            elif N == M == 3:
                return (Mat3x3, cls)
            elif M == N:
                return (Square, cls)
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
        shape_ = shape(args)
        try:
            nrows, ncols = shape_
        except ValueError:
            raise ValueError('invalid input shape: %s' % repr(shape_))
        if dtype is None:
            dtype = get_common_base(*[_dtype(x) for x in args])
        return cls[nrows, ncols, dtype](*args)

    def __init__(self, *args):
        self.flat = self._flat(sum(map(list, args), []), copy=False)

    @classmethod
    def fromdict(cls, D):
        '''Build a matrix from a dictionary from indexes to values'''

        N, M = cls.shape
        data = [0] * (N * M)
        for (i, j), value in D.items():
            if i >= cls.nrows or j >= cls.nrows:
                fmt = (N, M, (i, j)) 
                raise IndexError('invalid index for %sx%s matrix: %r' % fmt)
            k = i * N + j
            data[k] = value

        return cls.fromflat(data)

    @classmethod
    def fromrows(cls, rows, dtype=None):
        '''Build matrix from a sequence of row vectors'''

        # Obtain shape
        if cls.size is None:
            N = len(rows)
            M = len(rows[0])
        else:
            M, N = cls.shape
            
        # Collect data
        data = []
        for row in rows:
            if len(row) != M:
                raise ValueError('rows must be of same length')
            data.extend(row)

        # Choose the correct dtype
        if (dtype is None or dtype is cls.dtype) and cls.size is not None:
            return cls.fromflat(data)
        elif dtype is Any:
            return cls.fromflat(data, dtype=_dtype(data))
        else:
            T = (cls.__origin__ or cls)[N, M, dtype]
            return T.fromflat(data)

    @classmethod
    def fromcols(cls, cols):
        '''Build matrix from a sequence of column Vecs'''

        return cls.fromrows(cols).T

    #
    # Attributes
    #
    @property
    def T(self):
        return self.transpose()

    #
    # Return row Vecs or column Vecs
    #
    def asdict(self):
        '''Return matrix data as a map from indexes to (non-null)
        values.'''
        
        D = {}
        M, N = self.shape
        for i in range(M):
            for j in range(N):
                value = self[i, j]
                if value == 0:
                    D[i, j] = value
        return D  

    #
    # Iterators
    #
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

    def cols(self):
        '''Iterator over columns of matrix

        See also
        --------

        :meth:`rows`: iterate over the rows of a matrix.

        Example
        -------

        >>> M = Mat([1, 2], 
        ...         [3, 4])

        >>> list(M.cols())
        [Vec(1, 3), Vec(2, 4)]

        >>> list(M.rows())
        [Vec(1, 2), Vec(3, 4)]

        '''

        M = self.ncols
        data = self.flat
        for i in range(M):
            yield asvector(data[i::M])

    def rows(self):
        '''Iterator over row vectors'''

        return iter(self)

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

        >>> matrix.copy({(0, 0): 42, (0, 1): 3})
        Mat([42, 3], [3, 4])

        A similar operation can also be done by setting the corresponding
        keyword arguments. Remember that indexes start at zero as usual.

        >>> matrix.copy(A00=42, A01=3)
        Mat([42, 3], [3, 4])
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

        return self.fromflat(data)

    def transpose(self):
        '''Return the transposed matrix

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
        '''

        return self.fromrows(self.cols())


    #
    # Shape transformations
    #
    def __raise_badshape(self, data, s1):
        fmt = data, s1, self.nrows, self.nrows
        raise ValueError('incompatible %s size: %s in (%sx%s) matrix' % fmt)
    
    def withrow(self, data, index=None):
        '''Return a new matrix with and extra data inserted in the given index.
        If not index is given, insert data at the end.

        If the argument ``data`` is a matrix, multiple rows are inserted.'''

        N, M = self.shape
        data = list(self.flat)
        
        if isinstance(data, Mat):
            if data.ncols != N:
                self.__raise_badshape('row', data.ncols)
            newdata = data.flat
            M += data.nrows
        else:
            newdata = list(data)
            if len(newdata) != N:
                self.__raise_badshape('row', data.ncols)
            M += 1
            
        if index is None:
            data.extend(newdata)
        else:
            index = N * index
            data = data[:index] + newdata + data[index:]
            
        #FIXME: infer correct dtype
        return Mat[N, M, self.dtype].fromflat(data, copy=False)

    def withcol(self, data, index=None):
        '''Similar to M.withrow(), but works with columns'''

        N, M = self.shape
        cols = list(self.cols)
        if isinstance(data, Mat):
            if data.nrows != M:
                self.__raise_badshape('column', data.ncols)
            if index is None:
                cols.extend(data.cols)
            else:
                cols = cols[:index] + list(data.cols) + cols[index:]
        
        else:
            if index is None:
                cols.append(list(data))
            else:
                cols.insert(index, list(data))

        #FIXME: infer correct dtype
        return Mat[N, M, self.dtype].fromcols(data)

    def droppingcol(self, idx=None):
        '''Return a pair (matrix, col) with the new matrix with the extra column
        removed'''

        transpose, col = self.T.drop_row(idx)
        return transpose.T, col

    def droppingrow(self, idx=None):
        '''Return a pair (matrix, row) with the new matrix with the extra row
        removed'''

        if idx is None:
            idx = -1
        data = self.aslists()
        row = data.pop(idx)
        return self.fromrows(data), asvector(row)

    def selectcols(self, cols):
        '''Return a new matrix with the columns corresponding to the given
        sequence of indexes'''

        cols = tuple(cols)
        data = [[L[i] for i in cols] for L in self.aslists()]
        return self.fromrows(data)
    
    def selectrows(self, rows):
        pass

    #
    # Magic methods
    #
    def _fmt_number(self, x):
        # TODO: fix this to all matrices
        return ('%.3f' % x).rstrip('0').rstrip('.')

    def __repr__(self):
        data = ', '.join([repr(list(x)) for x in self])
        return '%s(%s)' % (self.__origin__.__name__, data)
        
    def __str__(self):
        N, matrix = self.shape
        fmt = self._fmt_number
        rows = [[fmt(x) for x in row] for row in self.rows()]
        sizes = sum([[len(x) for x in row] for row in rows], [])
        sizes_T = [sizes[i::N] for i in range(matrix)]
        sizes_max = [max(col) for col in sizes_T]
        rows = [[x.rjust(n) for n, x in zip(sizes_max, row)] for row in rows]
        rows = ['|%s|' % ('  '.join(row)) for row in rows]
        return '\n'.join(rows)

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

    #
    # Arithmetic operations
    #
    def __mul__(self, other):
        if isinstance(other, (Vec, tuple, list)):
            other = asvector(other)
            return asvector([u.dot(other) for u in self.rows()])

        elif isinstance(other, number):
            return self.fromflat([x * other for x in self.flat], copy=False)

        elif isinstance(other, Mat):
            cols = list(other.cols())
            rows = list(self.rows())
            data = sum([[u.dot(v) for u in cols] for v in rows], [])
            cls = self.__origin__[len(rows), len(cols), _dtype(data)]
            return cls.fromflat(data, copy=False)

        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, number):
            return self.fromflat([x * other for x in self.flat], copy=False)
        else:
            other = asvector(other)
            return asvector([u.dot(other) for u in self.cols()])

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
        return self.fromflat(x + y for (x, y) in self._zip_other(other))

    def __radd__(self, other):
        return self.fromflat(y + x for (x, y) in self._zip_other(other))

    def __sub__(self, other):
        return self.fromflat(x - y for (x, y) in self._zip_other(other))

    def __rsub__(self, other):
        return self.fromflat(y - x for (x, y) in self._zip_other(other))

    def __neg__(self):
        return self.fromflat(-x for x in self.flat)

    def __nonzero__(self):
        return True

#
# Matrix bases for specific shapes
#
class Square:
    '''
    Methods specific to square matrices
    '''
    
    @classmethod
    def fromdiag(cls, diag):
        '''Create diagonal matrix from diagonal terms'''

        N = len(diag)
        data = [0] * (N * N)
        for i in range(N):
            data[N * i + i] = diag[i]
        return cls.__origin__[N, N, _dtype(data)].fromflat(data, copy=False)

    def det(self):
        '''Return the determinant of the matrix'''

        raise NotImplementedError

    def trace(self):
        '''Computes the trace (i.e., sum of all elements in the diagonal)'''

        N = self.nrows
        return sum(i * N + i for i in range(N))

    def diag(self):
        '''Return a vector with the diagonal elements of the matrix'''

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
        '''Return a copy of the matrix with diagonal removed'''

        N = self.nrows
        data = list(self.flat)
        for i in range(N):
            data[i * N + i] *= 0
        return self._from_flat(data)

    def eig(self):
        '''Return a tuple of (eigenvalues, eigenvectors).'''

        return (self.eigval(), self.eigvec())

    def eigenvalues(self):
        '''Return a list of eigenvalues'''

        return [val for (val, _) in self.eigenpairs()]

    def eigenvectors(self):
        '''Return the matrix of normalized column eigenvectors.'''

        return self.fromcols([vec for (_, vec) in self.eigenpairs()])
    
    def eigenpairs(self):
        '''Return a list of (eigenvalue, eigenvector) pairs.'''
        
        return list(zip(self.eigval(), self.eigvec().cols()))

    def inv(self):
        '''Returns the inverse matrix'''

        # Simple and naive matrix inversion
        # Creates extended matrix
        N, M = self.shape
        dtype = promote_type(float, self.dtype)
        matrix = mMat[N, M, dtype].fromflat(self.flat)
        matrix = matrix.append_col(identity(N))

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
        return Mat[N, N, dtype].fromflat(out.flat)

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


class Mat2x2(Square):

    '''A 2 x 2 matrix'''


    __slots__ = ('_a', '_b', '_c', '_d')

    def __init__(self, row1, row2):
        dtype = self.dtype
        a, b = row1
        c, d = row2
        self._a = convert(a, dtype)
        self._b = convert(b, dtype)
        self._c = convert(c, dtype)
        self._d = convert(d, dtype)

    # Flat attribute handling
    @classmethod
    def fromflat(cls, data, copy=True, dtype=None):
        if cls.__concrete__ and dtype is None:
            a, b, c, d = data
            dtype = cls.dtype
            new = object.__new__(cls)
            new._a = convert(a, dtype)
            new._b = convert(b, dtype)
            new._c = convert(c, dtype)
            new._d = convert(d, dtype)
            return new
        return super().fromflat(data, copy=copy, dtype=dtype)
    
    @property
    def flat(self):
        return FlatView(self)
    
    def __flatiter__(self):
        yield self._a
        yield self._b
        yield self._c
        yield self._d
        
    def __flatgetitem__(self, i):
        if i == 0:
            return self._a
        elif i == 1:
            return self._b
        elif i == 2:
            return self._c
        elif i == 3:
            return self._d
        else:
            raise IndexError(i)

    def __flatsetitem__(self, i, value):
        if i == 0:
            self._a = convert(value, self.dtype)
        elif i == 1:
            self._b = convert(value, self.dtype) 
        elif i == 2:
            self._c = convert(value, self.dtype)
        elif i == 3:
            self._d = convert(value, self.dtype)
        else:
            raise IndexError(i)
    
    # Iterators
    def __iter__(self):
        vec = Vec[2, self.dtype]
        yield vec(self._a, self._b)
        yield vec(self._c, self._d)
    
    def cols(self):
        vec = Vec[2, self.dtype]
        yield vec(self._a, self._c)
        yield vec(self._b, self._d)

    def items(self):
        yield (0, 0), self._a
        yield (0, 1), self._b
        yield (1, 0), self._c
        yield (1, 1), self._d

    # Linear algebra operations
    def det(self):
        return self._a * self._d - self._b * self._c

    def trace(self):
        return self._a + self._d

    def diag(self):
        return Vec[2, self.dtype](self._a, self._d)

    def inv(self):
        det = self.det()
        return Mat([self._d / det, -self._b / det], 
                   [-self._c / det, self._a / det])
    
    def transpose(self):
        return type(self)([self._a, self._c],
                          [self._b, self._d])
    
    def eigenpairs(self):
        a, b, c, d = self.flat
        l1 = (d + a + self._sqrt(d * d - 2 * a * d + a * a + 4 * c * b)) / 2
        l2 = (d + a - self._sqrt(d * d - 2 * a * d + a * a + 4 * c * b)) / 2
        
        try:
            v1 = Vec(b / (l1 - a), 1)
        except ZeroDivisionError:
            v1 = Vec(1, 0)
        try:
            v2 = Vec(b / (l2 - a), 1)
        except ZeroDivisionError:
            v2 = Vec(1, 0)

        return [(l1, v1.normalized()), (l2, v2.normalized())]


class Mat3x3(object):

    def det(self):
        a, b, c, d, e, f, g, h, i = self.flat
        d1 = + (a * e * i)
        d2 = + (b * f * g)
        d3 = + (c * d * h)
        d4 = - (c * e * g)
        d5 = - (a * f * h)
        d6 = - (b * d * i)
        return d1 + d2 + d3 + d4 + d5 + d6

    def trace(self):
        return self.flat[0] + self.flat[4] + self.flat[8]

    def diag(self):
        return Vec[3, self.dtype](self.flat[0], self.flat[4], self.flat[8])
    
    def transpose(self):
        a, b, c, d, e, f, g, h, i = self.flat
        return self.fromflat([a, d, g,
                              b, e, h,
                              c, f, i], copy=False)

    def inv(self):
        Z = 1 / self.det() 
        a, b, c, d, e, f, g, h, i = self.flat
        data = [
            (e * i - f * h) * Z, (c * h - b * i) * Z, (b * f - c * e) * Z,
            (f * g - d * i) * Z, (a * i - c * g) * Z, (c * d - a * f) * Z,
            (d * h - e * g) * Z, (b * g - a * h) * Z, (a * e - b * d) * Z,
        ]
        return self.fromflat(data, copy=False)
    
    # TODO: use these faster versions
    def __mul_matrix(self, other):
        outtype = promote_type(type(self), type(other))
        a, b, c, d, e, f, g, h, i = self.flat
        j, k, l, m, n, o, p, q, r = other.flat
        data = [
            a * j + b * m + c * p, a * k + b * n + c * q, a * l + b * o + c * r,
            d * j + e * m + f * p, d * k + e * n + f * q, d * l + e * o + f * r,
            g * j + h * m + i * p, g * k + h * n + i * q, g * l + h * o + i * r,
        ]
        return outtype.fromflat(data, copy=False)

    def __mul_vector(self, other):
        a, b, c, d, e, f, g, h, i = self.flat
        x, y, z = other

        return Vec(a * x + d * y + g * z, 
                   b * x + e * y + h * z, 
                   c * x + f * y + i * z)

class mMat3x3:
    def itranspose(self):
        '''Transpose matrix *INPLACE*'''
        
        data = self.flat
        a, b, c = data[1], data[2], data[5]
        data[1], data[2] = data[3], data[6]
        data[3], data[5] = a, data[7]
        data[6], data[7] = b, c

#
# User-facing types
#
class Mat(MatAny, Immutable):

    '''A immutable matrix type'''


class mMat(MatAny, Mutable):
    
    '''A mutable version of Mat'''
    
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
            # FIXME: assert that mMat has a mutable flat
            self.flat._data[ki], self.flat._data[kj] = data[kj], data[ki]

    def irow_add(self, i, j, alpha=1):
        '''Adds alpha times row j to row i *inplace*'''

        matrix = self.ncols
        data = self.flat
        start_i = i * matrix
        start_j = j * matrix
        for r in range(matrix):
            ki = start_i + r
            kj = start_j + r
            
            # FIXME: assert mutable flat
            self.flat._data[ki] += alpha * data[kj]

    def irow_mul(self, i, value):
        '''Multiply row by the given value *inplace*'''

        matrix = self.ncols
        data = self.flat
        for j in range(matrix):
            # FIXME: assert mutable data
            self.flat._data[i * matrix + j] *= value

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


#
# Overloads and promotions
#
@overload(mul, (Mat, Vec))
def mul_matrix_Vec(M, v):
    return NotImplemented


@overload(mul, (Vec, Mat))
def mul_Vec_matrix(v, M):
    return NotImplemented
Vec._rotmatrix = Mat


#
# Matrices conversions
#   
def asmatrix(m):
    '''Return object as an immutable matrix'''

    if isinstance(m, Mat):
        return m
    else:
        return Mat(*m)
    
def asmmatrix(m):
    '''Return object as a mutable matrix'''

    if isinstance(m, mMat):
        return m
    else:
        return mMat(*m)


def asamatrix(m):
    '''Return object as an immutable matrix'''

    if isinstance(m, MatAny):
        return m
    else:
        return Mat(*m)
    

def identity(N, dtype=float):
    '''Return an identity matrix of size N by N'''

    return Mat[N, N, dtype].fromdiag([1] * N)


def midentity(N, dtype=float):
    '''Return a mutable identity matrix of size N by N'''

    return mMat[N, N, dtype].fromdiag([1] * N)


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
    print(repr(M))

    # print(matrix)
    #v = matrix.solve(u)
    #print(u)
    #print(matrix * v)
    #print(u)

    import doctest
    doctest.testmod()

    M2 = Mat.fromrows([[1, 2], [3, 4]])
    print(type(M2))