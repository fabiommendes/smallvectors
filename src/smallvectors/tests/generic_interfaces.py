from math import sqrt

import pytest


class LinearFixtures:
    base_cls = None

    @pytest.fixture
    def size(self):
        return self.base_cls.size

    @pytest.fixture
    def unity(self, size):
        return self.base_cls(*(1 / sqrt(size) for _ in range(size)))

    @pytest.fixture
    def null(self, size):
        return self.base_cls(*(0.0 for _ in range(size)))

    @pytest.fixture
    def u(self, size):
        return self.base_cls(*(x for x in range(1, size + 1)))

    @pytest.fixture
    def v(self, size):
        return self.base_cls(*reversed(self.u(size)))

    @pytest.fixture
    def e1(self, size):
        args = [0 for _ in range(size)]
        args[0] = 1
        return self.base_cls(*args)

    @pytest.fixture
    def e2(self, size):
        args = [0 for _ in range(size)]
        args[1] = 1
        return self.base_cls(*args)

    @pytest.fixture
    def e3(self, size):
        args = [0 for _ in range(size)]
        args[2] = 1
        return self.base_cls(*args)

    @pytest.fixture
    def e4(self, size):
        args = [0 for _ in range(size)]
        args[3] = 1
        return self.base_cls(*args)


class SequenceInterface:
    @pytest.fixture
    def size(self):
        return self.base_cls.size

    def test_has_basic_sequence_interface(self, u, size):
        assert len(u) == size

    def test_equality_with_non_tuple_sequences(self, u):
        assert u != '12'
        assert u != None
        assert u != list(u) + [0]

    def test_can_get_item(self, u):
        assert u[0] is not None
        assert u[-1] is not None

    def test_can_slice(self, u):
        assert u[:] == list(u)

    def test_index_error_for_invalid_indexes(self, u):
        n = len(u)
        with pytest.raises(IndexError):
            u[n]
        with pytest.raises(IndexError):
            u[-(n + 1)]

    def test_copy_is_equal_to_itself(self, u):
        assert u == u.copy()