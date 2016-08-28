from generic import promote_type

from smallvectors.tools import dtype as _dtype
from smallvectors.vector import asvector


class SquareMixin:
    """
    Methods specific to square matrices
    """

    @classmethod
    def from_diag(cls, diag):
        """
        Create diagonal matrix from diagonal terms.
        """

        N = len(diag)
        data = [0] * (N * N)
        for i in range(N):
            data[N * i + i] = diag[i]
        return cls.__origin__[N, N, _dtype(data)].from_flat(data, copy=False)

    def det(self):
        """
        Return the determinant of the matrix.
        """

        raise NotImplementedError

    def trace(self):
        """
        Computes the trace (i.e., sum of all elements in the diagonal).
        """

        N = self.nrows
        data = self.flat
        return sum(data[i * N + i] for i in range(N))

    def diag(self):
        """
        Return a vector with the diagonal elements of the matrix.
        """

        N = self.nrows
        data = self.flat
        return asvector([data.flat[i * N + i] for i in range(N)])

    def set_diag(self, diag):
        """
        Return a copy of the matrix with different diagonal terms.
        """

        M = self.nrows
        if len(diag) != M:
            raise ValueError('wrong size for diagonal')

        # Write diagonal in a copy of flat data
        data = list(self.flat)
        for i, x in enumerate(diag):
            data[i * M + i] = x

        return self.from_flat(data, copy=False)

    def drop_diag(self):
        """
        Return a copy of the matrix with diagonal removed (all elements are
        set to zero).
        """

        N = self.nrows
        data = list(self.flat)
        for i in range(N):
            data[i * N + i] *= 0
        return self.from_flat(data, copy=False)

    def eig(self):
        """
        Return a tuple of (eigenvalues, eigenvectors).
        """

        return (self.eigval(), self.eigvec())

    def eigenvalues(self):
        """
        Return a list of eigenvalues.
        """

        return [val for (val, _) in self.eigenpairs()]

    def eigenvectors(self):
        """
        Return the matrix of normalized column eigenvectors.
        """

        return self.fromcols([vec for (_, vec) in self.eigenpairs()])

    def eigenpairs(self):
        """
        Return a list of (eigenvalue, eigenvector) pairs.
        """

        return list(zip(self.eigval(), self.eigvec().cols()))

    def inv(self):
        """
        Return the inverse matrix.
        """

        # Simple and naive matrix inversion using Gaussian elimination
        # Creates extended matrix
        N = self.nrows
        dtype = promote_type(float, self.dtype)
        matrix = self._mmatrix[N, N, dtype].from_flat(self.flat)
        matrix = matrix.append_col(self._identity(N))

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
                matrix.isum_row(k, matrix[i] * (-matrix[k, i] / Z))

            # Make the left hand side diagonal
            Z = matrix[i, i]
            for k in range(0, i):
                matrix.isum_row(k, matrix[i] * (-matrix[k, i] / Z))

        # Normalize by the diagonal
        for i in range(N):
            matrix.imul_row(i, 1 / matrix[i, i])

        out = matrix.select_cols(range(N, 2 * N))
        return self._matrix[N, N, dtype].from_flat(out.flat)

    def solve(self, b, method='gauss', **kwds):
        """
        Solve the linear system ``matrix * x  = b`` for x.
        """

        method = getattr(self, 'solve_%s' % method)
        return method(b, **kwds)

    def solve_jacobi(self, b, tol=1e-3, x0=None, maxiter=1000):
        """
        Solve a linear using Gauss-Jacobi method.
        """

        b = asvector(b)
        x = b * 0
        D = self.from_diag([1.0 / x for x in self.diag()])
        R = self.droppingdiag()

        for _ in range(maxiter):
            x, old = D * (b - R * x), x
            if (x - old).norm() < tol:
                break
        return x

    def solve_gauss(self, b):
        """
        Solve system by simple Gaussian elimination.
        """

        # Creates extended matrix
        N = self.nrows
        matrix = self._mmatrix[N, N, self._floating].from_flat(self.flat)
        matrix = matrix.append_col(b)

        for i in range(0, N - 1):
            # Search for maximum value in the truncated column and put it
            # in pivoting position
            trunc_col = matrix.col(i)[i:]
            _, idx = max([(abs(c), i) for (i, c) in enumerate(trunc_col)])
            matrix.iswap_rows(i, idx + i)

            # Find linear combinations that make all rows below the current one
            # become equal to zero in the current column
            Z = matrix[i, i]
            for k in range(i + 1, N):
                matrix.isum_row(k, matrix[i] * (-matrix[k, i] / Z))

        # Solve equation Ax=b for an upper triangular matrix
        A, b = matrix.drop_col()
        return A.solve_triangular(b)

    def solve_triangular(self, b, lower=False):
        """
        Solve a triangular system.

        If lower=True, it assumes a lower triangular matrix, otherwise (default)
        assumes an upper triangular matrix.
        """

        N = self.nrows
        x = [0] * N
        if lower:
            for i in range(N):
                x[i] = (b[i] - self[i].dot(x)) / self[i, i]
        else:
            for i in range(N - 1, -1, -1):
                x[i] = (b[i] - self[i].dot(x)) / self[i, i]
        return asvector(x)
