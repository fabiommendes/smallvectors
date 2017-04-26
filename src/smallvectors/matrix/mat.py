import abc

from smallvectors import Vec, asvector
from smallvectors.core import SmallVectorsBase, \
    AddElementWise, MulScalar
from smallvectors.utils import flatten, dtype as _dtype

number = (float, int)


class Mat(metaclass=abc.ABCMeta):
    """
    Base class for immutable matrix types
    """

    __slots__ = ()
    __parameters__ = (int, int, type)
    nrows = ncols = None
    size = None
    shape = None
    _square_base = None
    _mat2x2_base = None
    _mat3x3_base = None

    @property
    def T(self):
        """
        Matrix transpose.
        """

        return self.transposed()

    @property
    def H(self):
        """
        Hermitian transpose.
        """
        return self.transposed().conjugate()

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
    def from_rows(cls, rows):
        """
        Build matrix from a sequence of row vectors.
        """

        data, M, N = flatten(rows, 2)
        if (M, N) != cls.shape:
            if cls.size is None:
                return cls.from_flat(data)
            msg = ('data shape %s is not consistent with matrix type %s' %
                   ((M, N), cls.shape))
            raise ValueError(msg)
        return cls.from_flat(data)

    @classmethod
    def from_cols(cls, cols):
        """
        Build matrix from a sequence of column vectors.
        """

        dataT, N, M = flatten(cols, 2)
        data = dataT[:]
        for i in range(M):
            data[i * N:i * N + N] = dataT[i::M]

        if (M, N) != cls.shape:
            if cls.size is None:
                return cls.from_flat(data)
            msg = ('data shape %s is not consistent with matrix type %s' %
                   ((M, N), cls.shape))
            raise ValueError(msg)
        return cls.from_flat(data)

    def __repr__(self):
        data = ', '.join([repr(list(x)) for x in self])
        return '%s(%s)' % (self.__class__.__name__, data)

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
            return self.from_flat([x * other for x in self.flat])

        elif isinstance(other, Mat):
            cols = list(other.cols())
            rows = list(self.rows())
            data = sum([[u.dot(v) for u in cols] for v in rows], [])
            return type(self).from_flat(data)

        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, number):
            return self.from_flat([x * other for x in self.flat])
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

    def __add__(self, other):
        try:
            return self.from_flat(x + y for (x, y) in self._zip_other(other))
        except NotImplementedError:
            return NotImplemented

    def __radd__(self, other):
        try:
            return self.from_flat(y + x for (x, y) in self._zip_other(other))
        except NotImplementedError:
            return NotImplemented

    def __sub__(self, other):
        try:
            return self.from_flat(x - y for (x, y) in self._zip_other(other))
        except NotImplementedError:
            return NotImplemented

    def __rsub__(self, other):
        try:
            return self.from_flat(y - x for (x, y) in self._zip_other(other))
        except NotImplementedError:
            return NotImplemented

    def __neg__(self):
        return self.from_flat(-x for x in self.flat)

    def __nonzero__(self):
        return True

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
                if value != 0:
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
            :method:`rows`: iterate over the rows of a matrix.

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

    def copy(self, D=None, **kwargs):
        """
        Return a copy of matrix possibly changing some terms.

        Can set terms in the matrix by either passing a mapping from (i, j)
        indexes to values or by passing keyword arguments of the form
        ``Aij=value``. The keyword form only accept single digit indexes

        Example:
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
        if kwargs:
            D = dict(D or {})
            for k, v in kwargs.items():
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

    def transposed(self):
        """
        Return the transposed matrix

        Example:
            Transpose mirrors items around the diagonal.

            >>> matrix = Mat([1, 2], [3, 4])
            >>> print(matrix)
            |1  2|
            |3  4|

            >>> print(matrix.transposed())
            |1  3|
            |2  4|

            ``matrix.T`` is an alias to the transpose() method

            >>> matrix.T == matrix.transposed()
            True
        """

        N, M = self.shape
        if N == M:
            return self.from_cols(self.rows())
        else:
            return type(self).from_cols(self.rows())

    # Shape transformations
    def __raise_bad_shape(self, data, s1):
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
                self.__raise_bad_shape('row', data.ncols)
            extra = list(data.flat)
            N += data.nrows
        else:
            extra = list(data)
            if len(data) != M:
                self.__raise_bad_shape('row', len(data))
            N += 1

        if index is None:
            flat.extend(extra)
        else:
            index = M * index
            flat = flat[:index] + extra + flat[index:]
        return Mat.from_flat(flat)

    def append_col(self, data, index=None):
        """
        Similar to M.append_row(), but works with columns.
        """

        N, M = self.shape
        cols = list(self.cols())
        if isinstance(data, Mat):
            if data.nrows != N:
                self.__raise_bad_shape('column', data.ncols)
            if index is None:
                cols.extend(data.cols())
            else:
                cols = cols[:index] + list(data.cols()) + cols[index:]
            M += len(data)

        else:
            if len(data) != N:
                self.__raise_bad_shape('row', len(data))
            if index is None:
                cols.append(list(data))
            else:
                cols.insert(index, list(data))
            M += 1

        return Mat.from_cols(cols)

    def drop_col(self, idx=-1):
        """
        Return a pair (matrix, col) with the new matrix with the extra column
        removed.

        If no index is given, drops the last column.
        """

        M, N = self.shape
        data = list(self.cols())
        col = data.pop(idx)
        return Mat.from_cols(data), col

    def drop_row(self, idx=-1):
        """
        Return a pair (matrix, row) with the new matrix with the extra row
        removed.

        If no index is given, drops the last row.
        """

        M, N = self.shape
        data = list(self.rows())
        row = data.pop(idx)
        return Mat.from_rows(data), row

    def select_cols(self, *indexes):
        """
        Return a new matrix with the columns with given indexes.
        """

        L = list(self.cols())
        data = [L[i] for i in indexes]
        return Mat.from_cols(data)

    def select_rows(self, *indexes):
        """
        Return a new matrix with the rows corresponding to the given
        sequence of indexes.
        """

        L = list(self.rows())
        data = [L[i] for i in indexes]
        return Mat.from_rows(data)

    def _zip_other(self, other):
        """
        Auxiliary function that zip the flat iterator of both operands if
        the shapes are valid.
        """

        if not isinstance(other, Mat):
            raise NotImplementedError

        if (self.ncols != other.ncols) or (self.nrows != other.nrows):
            n, m = self.shape
            a, b, = other.shape
            msg = 'not aligned: [%s x %s] to [%s x %s]' % (n, m, a, b)
            raise ValueError(msg)

        return zip(self.flat, other.flat)


