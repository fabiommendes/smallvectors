#
# Cython accelerated 2D vectors
#

cpdef class Linear2Dlf:
    cdef double _data[2]

    def __cinit__(self, x, y):
        pass


cpdef class Vec2Dlf(Linear2Dlf):
    def __add__(self, other):
        cdef Vec2Dlf other_vec
        try:
            other_vec = other
            return Vec2Dlf(other_vec._data[0] + self._data[0],
                           other_vec._data[1] + self._data[1])
        except TypeError:
            return super().__add__(other)
