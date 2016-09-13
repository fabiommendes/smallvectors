import pytest

from smallvectors import simeq


class TestBinaryOperation:
    a_alts = ()
    b_alts = ()
    base_args__zero = ()
    base_kwargs__zero = {}
    base_scalar = 2

    @pytest.fixture
    def scalar(self):
        return self.base_scalar

    @pytest.fixture
    def one(self):
        return 1

    @pytest.fixture
    def scalar_zero(self):
        return 0

    @pytest.fixture
    def a(self, obj):
        return obj

    @pytest.fixture
    def b(self, other):
        return other

    @pytest.fixture
    def zero(self, cls):
        return cls(*self.base_args__zero, **self.base_kwargs__zero)


class TestPairwiseAddition(TestBinaryOperation):
    base_args__add_ab = ()
    base_kwargs__add_ab = {}
    base_args__sub_ab = ()
    base_kwargs__sub_ab = {}

    @pytest.fixture
    def add_ab(self, cls):
        return cls(*self.base_args__add_ab, **self.base_kwargs__sub_ab)

    @pytest.fixture
    def sub_ab(self, cls):
        return cls(*self.base_args__sub_ab, **self.base_kwargs__sub_ab)

    def test_addition(self, a, b, add_ab):
        assert a + b == add_ab

    def test_add_alternatives(self, a, b, add_ab):
        for alt in self.a_alts:
            assert alt + b == add_ab
        for alt in self.b_alts:
            assert a + alt == add_ab

    def test_sub(self, a, b, sub_ab):
        assert a - b == sub_ab

    def test_sub_alternatives(self, a, b, sub_ab):
        for alt in self.a_alts:
            assert alt - b == sub_ab
        for alt in self.b_alts:
            assert a + alt == sub_ab

    def test_addition_is_commutative(self, a, b):
        assert a + b == b + a

    def test_sub_is_not_commutative(self, a, b):
        assert a - b != b - a

    def test_does_not_contribute_to_addition(self, a, b, zero):
        assert a + zero == a
        assert zero + a == a
        assert b + zero == b
        assert zero + b == b
        assert zero + zero == zero

    def test_does_not_contribute_to_subtraction(self, a, b, zero):
        assert a - zero == a
        assert b - zero == b
        assert zero - zero == zero

    def test_cannot_add_scalar(self, a, scalar):
        with pytest.raises((ValueError, TypeError)):
            a + scalar
        with pytest.raises((ValueError, TypeError)):
            a - scalar
        with pytest.raises((ValueError, TypeError)):
            scalar + a
        with pytest.raises((ValueError, TypeError)):
            scalar - a


class TestPairwiseMultiplication(TestBinaryOperation):
    base_args__mul = ()

    def test_pairwise_multiplication(self, a, b, mul):
        assert simeq(a * b, mul)

    def test_multiplication_by_zero(self, a, b, zero):
        assert a * zero == zero
        assert b * zero == zero
        assert zero * a == zero
        assert zero * b == zero


class TestScalarMultiplication(TestBinaryOperation):
    base_args__smul = ()

    def test_scalar_multiplication(self, obj, scalar, smul):
        assert simeq(obj * scalar, smul)
        assert simeq(scalar * obj, smul)

    def test_multiplication_by_one(self, a, b, one):
        assert one * a == a
        assert a * one == a
        assert one * b == b
        assert b * one == b

    def test_multiplication_by_zero(self, a, b, zero, scalar_zero):
        assert scalar_zero * a == zero
        assert a * scalar_zero == zero
        assert scalar_zero * b == zero
        assert b * scalar_zero == zero

    def test_scalar_division(self, obj, scalar, smul):
        value = 1 / scalar
        assert simeq(obj / value, smul)

    def test_division_by_one(self, obj, one):
        assert obj / one == obj
        assert obj // one == obj

    def test_rhs_division_fails(self, a, one):
        with pytest.raises((ValueError, TypeError)):
            one / a


