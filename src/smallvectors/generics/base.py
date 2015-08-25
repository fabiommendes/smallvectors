from generic import generic, overload

#
# Linear algebra functions
#


@generic
def norm(x):
    return x.norm()
