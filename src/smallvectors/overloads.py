'''
A module carrying overloadable generic functions for the smallvectors package.
'''

def dot(u, v):
    '''Calcula o produto escalar entre dois vetores

    Exemplo
    -------

    >>> dot((1, 2), (3, 4))
    11
    '''

    try:
        return u.dot(v)
    except AttributeError:
        pass

    try:
        return v.dot(u)
    except AttributeError:
        pass

    if len(u) == len(v):
        return sum(x * y for (x, y) in zip(u, v))
    else:
        raise ValueError('length mismatch: %s and %s')


def cross(v1, v2):
    '''Retorna a compontente z do produto vetorial de dois vetores
    bidimensionais'''

    x1, y1 = v1
    x2, y2 = v2
    return x1 * y2 - x2 * y1


def diag(a, b):
    '''Retorna uma matrix com os valores de a e b na diagonal principal'''

    return Mat2([[a, 0], [0, b]])
