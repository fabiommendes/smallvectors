from .vectors import Vec

#
# Direction
#
class Direction(Vec):

    '''Direction is an immutable Vec with unitary length and represents a
    direction in euclidian space'''

    __slots__ = ()

    def is_null(self):
        '''Always False for Direction objects'''

        return False

    def is_unity(self, tol=1e-6):
        '''Always True for Direction objects'''

        return True

    def norm(self):
        '''Returns the norm of a vector'''

        return 1.0

    def norm_sqr(self):
        '''Returns the squared norm of a vector'''

        return 1.0

    def normalized(self):
        '''Return a normalized version of vector'''

        return self

class Direction2D:

    '''A 2-dimensional direction/unity vector'''

    __slots__ = ()

    def __init__(self, x, y):
        norm = self._sqrt(x * x + y * y)
        if norm == 0:
            raise ValueError('null vector does not define a valid direction')

        self._x = x / norm
        self._y = y / norm

    def rotated(self, theta):
        '''Rotate vector by an angle theta around origin'''

        x, y = self
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        new = Vec2D.__new__(Direction2, x, y)
        new.x = x * cos_t - y * sin_t
        new.y = x * sin_t + y * cos_t
        return new



#
# Direction conversions
#
def asdirection(v):
    '''Return the argument as a Direction instance.'''

    if isinstance(v, Direction):
        return v
    else:
        return Direction(*v)