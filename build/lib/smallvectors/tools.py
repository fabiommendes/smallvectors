# -*- coding: utf8 -*-
from smallvectors import Vec, Point, Direction


def sign(x):
    '''Returns the sign of a number'''

    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def shadow_x(A, B):
    '''Overlap between shapes A and B in the x direction'''

    return min(A.xmax, B.xmax) - max(A.xmin, B.xmin)


def shadow_y(A, B):
    '''Retorna o tamanho da sombra comum entre os objetos A e B no eixo y.
    Caso não haja superposição, retorna um valor negativo que corresponde ao
    tamanho do buraco'''

    return min(A.ymax, B.ymax) - max(A.ymin, B.ymin)


def asvector(v):
    '''Return the argument as a Vec instance.'''

    if isinstance(v, Vec):
        return v
    else:
        return Vec.from_seq(v)


def aspoint(v):
    '''Return the argument as a Point instance.'''

    if isinstance(v, Point):
        return v
    else:
        return Point.from_seq(v)


def asdirection(v):
    '''Return the argument as a Direction instance.'''

    if isinstance(v, Direction):
        return v
    else:
        return Direction.from_seq(v)


# TODO: move to smallvectors
def dot(v1, v2):
    '''Calcula o produto escalar entre dois vetores

    Exemplo
    -------

    >>> dot((1, 2), (3, 4))
    11

    >>> dot(Vec2(1, 2), Vec2(3, 4))
    11.0
    '''

    try:
        A = v1
        B = v2
        return A.x * B.x + A.y * B.y
    except (AttributeError, TypeError):
        return sum(x * y for (x, y) in zip(v1, v2))


def cross(v1, v2):
    '''Retorna a compontente z do produto vetorial de dois vetores
    bidimensionais'''

    try:
        A = v1
        B = v2
        x1 = A.x
        y1 = A.y
        x2 = B.x
        y2 = B.x
    except (AttributeError, TypeError):
        x1, y1 = v1
        x2, y2 = v2
    return x1 * y2 - x2 * y1


def asmatrix(m):
    '''Retorna o objeto como uma instância da classe Vetor'''

    if isinstance(m, Mat2):
        return m
    else:
        return Mat2(m)


def diag(a, b):
    '''Retorna uma matrix com os valores de a e b na diagonal principal'''

    return Mat2([[a, 0], [0, b]])

if __name__ == '__main__':
    import doctest
    doctest.testmod()
