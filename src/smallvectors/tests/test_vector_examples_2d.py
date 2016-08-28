from smallvectors import Vec
from smallvectors.tests.test_arithmetic import *


make_arithmetic_fixtures(
    zero=Vec(0, 0),
    x=Vec(1, 2), x_alts=[[1, 2], (1, 2)],
    y=Vec(3, 4), y_alts=[[3, 4], (3, 4)],
    scalar=2.0, scalar_add=False, smul=Vec(2, 4),
    add=Vec(4, 6), sub=Vec(-2, -2),
    ns=globals(),
)


def test_invalid_add_scalar(x):
    with pytest.raises(TypeError):
        y = x + 1


def test_invalid_sub_scalar(x):
    with pytest.raises(TypeError):
        y = x - 1


def test_invalid_mul_tuple(x):
    with pytest.raises(TypeError):
        y = x * (1, 2)


def test_invalid_mul_vec(x):
    with pytest.raises(TypeError):
        y = x * x


def test_invalid_div_tuple(x):
    with pytest.raises(TypeError):
        y = x / (1, 2)
    with pytest.raises(TypeError):
        y = (1, 2) / x


def test_invalid_div_vec(x):
    with pytest.raises(TypeError):
        y = x / x


def test_invalid_div_scalar(x):
    with pytest.raises(TypeError):
        y = 1 / x

#
# Tests vector API
#
def test_rotated_is_new(x):
    assert x.rotate(0.0) is not x


def test_rotated_keeps_norm(x):
    for t in range(20):
        Z1 = x.norm()
        Z2 = x.rotate(6.28 * t / 20).norm()
        assert abs(Z1 - Z2) < 1e-6, (Z1, Z2)


def test_has_2d_methods(x):
    for attr in ['perp', 'from_polar']:
        assert hasattr(x, attr)


def test_vec_almost_equal(x, y):
    y = x + y / 1e9
    w = x + 0 * y
    assert x.almost_equal(y)
    assert y.almost_equal(x)
    assert x.almost_equal(w)
    assert w.almost_equal(x)


if __name__ == '__main__':
    pytest.main('test_vector.py -q')
