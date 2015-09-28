from ..core import Mutable, Immutable
from . import VecOrPoint


class AnyPoint(VecOrPoint):
    pass


class Point(AnyPoint, Immutable):

    '''Immutable point type'''


class mPoint(AnyPoint, Mutable):

    '''A mutable point type'''


if __name__ == '__main__':
    from smallvectors.cartesian.anyvec import Vec
    from smallvectors.generics import add

    p = Point(1, 2)
    v = Vec(1, 2)

    print(p + [1, 2])

    print(add[Vec, Point](p, v))
    print(p + v)
