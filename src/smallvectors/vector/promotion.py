import operator

from generic import set_promotion, set_conversion
from generic.op import add, sub

from smallvectors.vector import Vec, mVec


@set_promotion(Vec, Vec)
def promote_vectors(u, v):
    """Promote two Vec types to the same type"""

    u_type = type(u)
    v_type = type(v)
    if u_type is v_type:
        return (u, v)

    # Test shapes
    if u_type.shape != v_type.shape:
        raise TypeError('vectors have different shapes')

    # Fasttrack common cases
    u_dtype = u.dtype
    v_dtype = v.dtype
    if u_dtype is float and v_dtype is int:
        return u, v.convert(float)
    elif u_dtype is int and v_dtype is float:
        return u.convert(float), v

    zipped = [promote(x, y) for (x, y) in zip(u, v)]
    u = Vec(*[x for (x, y) in zipped])
    v = Vec(*[y for (x, y) in zipped])
    return u, v


# Conversions and promotions between vec types and tuples/lists
set_conversion(Vec, tuple, tuple)
set_conversion(Vec, list, list)
for T in [Vec, mVec]:
    set_conversion(tuple, T, T)
    set_conversion(list, T, T)


@set_promotion(Vec, tuple, symmetric=True, restype=Vec)
@set_promotion(Vec, list, symmetric=True, restype=Vec)
def promote(u, v):
    return u, u.__origin__.from_seq(v)


def asvector_overload(op, tt):
    real_op = getattr(operator, op.__name__)

    @op.overload((Vec, tt))
    def overload(u, v):
        return real_op(u, Vec(*v))

    @op.overload((tt, Vec))
    def overload(u, v):  # @DuplicatedSignature
        return real_op(Vec(*u), v)


for op in [add, sub]:
    for tt in [tuple, list]:
        asvector_overload(op, tt)
