import sys


def default_smallvectors_type_ns(params):
    """
    Compute a dict with shape, ndim, dtype and size attributes computed from
    the given parameters.
    """

    ns = {}
    if params is None:
        return ns

    shape = list(params[:-1])
    for i, x in enumerate(shape):
        if x is int:
            shape[i] = None
    ns['shape'] = shape = tuple(shape)
    ns['ndim'] = len(shape)

    dtype = params[-1]
    if dtype is type:
        dtype = None
    ns['dtype'] = dtype

    size = 1
    for dim in shape:
        if dim is None:
            ns['size'] = None
            break
        size *= dim
    else:
        ns['size'] = size
    return ns


def get_sibling_classes(cls):
    """
    Helper function for finding the Mutable/Immutable pair of classes.
    """

    parent = cls.mro()[1]
    mod = sys.modules[cls.__module__]
    names = (x for x in dir(mod) if not x.startswith('_'))
    objects = (getattr(mod, x) for x in names)
    types = (x for x in objects if isinstance(x, type))
    siblings = [T for T in types if issubclass(T, parent)]
    return siblings