"""
This is somewhat a meta-test: we define some tests for arithmetic operations
and test them on int's to validate.

Other types may reuse many of these functions by importing from this module.
"""
import pytest


def make_arithmetic_fixtures(
        x, y, zero,
        scalar=2,
        scalar_add=True, scalar_mul=True,
        elem_add=True, elem_mul=False,
        x_alts=[], y_alts=[],
        add=None, sub=None, mul=None, div=None,
        radd=None, rsub=None, rmul=None, rdiv=None,
        sadd=None, smul=None,
        simeq=lambda x, y: abs(x - y) < 1e-3,
        ns=None,
    ):
    """Make fixtures from arguments and insert them in module's namespace"""

    if radd is None and elem_add: radd = add
    if rmul is None and elem_mul: rmul = mul
    if rsub is None and elem_add: rsub = -sub
    if rdiv is None and elem_mul: rdiv = 1 / div
    kwds = locals()

    # Create fixtures
    def make_echo(a):
        def echo():
            return a
        return echo

    for k, v in kwds.items():
        fixture = make_echo(v)
        fixture.__name__ = k
        kwds[k] = pytest.fixture(fixture)
    ns.update(kwds)


make_arithmetic_fixtures(x=1, y=2, zero=0, scalar=2.0,
              add=3, sub=-1, mul=2, div=0.5, rdiv=2,
              sadd=3.0, smul=2.0,
              x_alts=[1.0], y_alts=[2.0],
              ns=globals())


# Same type operations
def test_add(x, y, zero, add, radd, elem_add):
    if elem_add:
        assert x + y == add
        assert y + x == radd
        assert x + zero == zero + x == x
        assert y + zero == zero + y == y


def test_sub(x, y, zero, sub, rsub, elem_add):
    if elem_add:
        assert x - y == sub
        assert y - x == rsub
        assert x - zero == x
        assert y - zero == y


def test_mul(x, y, mul, rmul, elem_mul):
    if elem_mul:
        assert x * y == mul
        assert y * x == rmul


def test_div(x, y, div, rdiv, elem_mul):
    if elem_mul:
        assert x / y == div
        assert y / x == rdiv


# Scalar operations
def test_add_scalar(x, scalar, sadd, scalar_add):
    if scalar_add:
        assert x + scalar == sadd
        assert scalar + x == sadd


def test_sub_scalar(x, scalar, sadd, scalar_add):
    if scalar_add:
        assert (-x) - scalar == -sadd


def test_mul_scalar(x, zero, scalar, smul, scalar_mul):
    if scalar_mul:
        assert x * scalar == smul
        assert scalar * x == smul
        assert 0 * x == x * 0 == zero
        assert 1 * x == x * 1 == x


def test_div_scalar(x, scalar, smul, scalar_mul):
    if scalar_mul:
        assert x / (1 / scalar) == smul
        assert x / 1 == x


# Operations with alternatives
def test_add_alts(x, y, x_alts, y_alts, add, radd, elem_add):
    if elem_add:
        for x_alt in x_alts:
            assert x_alt + y == add
            assert y + x_alt == radd
        for y_alt in y_alts:
            assert x + y_alt == add
            assert y_alt + x == radd


def test_sub_alts(x, y, x_alts, y_alts, sub, rsub, elem_add):
    if elem_add:
        for x_alt in x_alts:
            assert x_alt - y == sub
            assert y - x_alt == rsub
        for y_alt in y_alts:
            assert x - y_alt == sub
            assert y_alt - x == rsub


def test_mul_alts(x, y, x_alts, y_alts, mul, rmul, elem_mul):
    if elem_mul:
        for x_alt in x_alts:
            assert x_alt * y == mul
            assert y * x_alt == rmul
        for y_alt in y_alts:
            assert x * y_alt == mul
            assert y_alt * x == rmul


def test_div_alts(x, y, x_alts, y_alts, div, rdiv, elem_mul):
    if elem_mul:
        for x_alt in x_alts:
            assert x_alt / y == div
            assert y / x_alt == rdiv
        for y_alt in y_alts:
            assert x / y_alt == div
            assert y_alt / x == rdiv


def test_equal_alternatives(x, y, x_alts, y_alts):
    for x_alt in x_alts:
        assert x_alt == x
        assert x == x_alt
    for y_alt in y_alts:
        assert y_alt == y
        assert y == y_alt


__all__ = ['make_arithmetic_fixtures', 'pytest']
__all__.extend(name for name in globals() if name.startswith('test_'))


if __name__ == '__main__':
    pytest.main('test_arithmetic.py -q')
