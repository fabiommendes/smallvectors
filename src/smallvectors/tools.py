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

# TODO: move to smallvectors

if __name__ == '__main__':
    import doctest
    doctest.testmod()
