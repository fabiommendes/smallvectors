import pytest

from smallvectors import Immutable
from smallvectors.core.mutability import Mutable, Immutable


@pytest.fixture
def tol():
    return 1e-5


def abstract_prop(name):
    @property
    def fget(self):
        raise NotImplementedError('please define %r in your test class' % name)

    return fget


def assert_triangular_identity(a, b, norm):
    assert (a + b).norm(norm) <= a.norm(norm) + b.norm(norm)


class TestClass:
    """
    Tests around a base class.
    """

    base_cls = abstract_prop('base_cls')
    base_args = abstract_prop('base_args')

    @pytest.fixture
    def obj(self):
        return self.base_cls(*self.base_args)

    @pytest.fixture
    def cls(self):
        return self.base_cls

    @pytest.fixture
    def args(self):
        return self.base_args

    @pytest.fixture
    def tol(self):
        return 1e-5

    def random_obj(self):
        """
        A random object.
        """

        return self.base_cls(*self.random_args())

    def random_args(self):
        """
        A random valid argument used to instantiate a random object
        """

        raise NotImplementedError

    def test_object_has_no_dict(self, obj):
        cls_list = list(type(obj).mro())[:-1]
        for subclass in cls_list:
            assert '__slots__' in subclass.__dict__, \
                '%s has no slots' % subclass

        with pytest.raises(AttributeError):
            obj.not_a_valid_attribute_name = None


class TestMutability(TestClass):
    """
    Tests the mutable/immutable interface
    """

    @property
    def immutable_cls(self):
        return self.base_cls

    @property
    def mutable_cls(self):
        return self.base_cls.__mutable_class__

    @pytest.fixture
    def mutable(self):
        return self.mutable_cls(*self.base_args)

    @pytest.fixture
    def immutable(self):
        return self.immutable_cls(*self.base_args)

    def test_classes_define_mutability(self):
        assert issubclass(self.mutable_cls, Mutable)
        assert issubclass(self.immutable_cls, Immutable)

    def test_construct_mutable(self):
        self.mutable_cls(*self.base_args)

    def test_construct_immutable(self):
        self.immutable_cls(*self.base_args)

    def test_mutable_equality(self, mutable):
        new = self.mutable_cls(*self.base_args)
        assert mutable == new

    def test_immutable_equality(self, immutable):
        new = self.immutable_cls(*self.base_args)
        assert immutable, new

    def test_mutable_equals_to_immutable(self, mutable, immutable):
        assert mutable == immutable
        assert immutable == mutable

    def test_is_immutable(self, immutable):
        assert isinstance(immutable, Immutable)
        assert not isinstance(immutable, Mutable)
        assert immutable.is_immutable() is True
        assert immutable.is_mutable() is False

    def test_is_mutable(self, mutable):
        assert isinstance(mutable, Mutable)
        assert not isinstance(mutable, Immutable)
        assert mutable.is_mutable() is True
        assert mutable.is_immutable() is False


class DisableMutabilityTests:
    """
    Disable mutability tests.

    This is useful to inherit test cases that test mutability in a situation
    in which mutability tests are not desirable.
    """

    base_cls = None

    @property
    def mutable_cls(self):
        return self.base_cls

    @property
    def immutable_cls(self):
        return self.base_cls

for _attr, _method in vars(TestMutability).items():
    if _attr.startswith('test_') and not hasattr(DisableMutabilityTests, _attr):
        setattr(DisableMutabilityTests, _attr, None)


class TestFlatable(TestClass):
    """
    Tests the Flatable interface.
    """

    def test_flatable_expansion(self, obj):
        assert list(obj.flat)

    def test_flatable_expansion_equivalence(self, obj):
        flat = list(obj.flat)
        new = self.base_cls.from_flat(flat)
        assert new == obj

    def test_flatable_length(self, obj):
        assert len(obj.flat) >= 0

    def test_flatable_getitem(self, obj):
        L = list(obj.flat)
        for i in range(len(obj.flat)):
            assert L[i] == obj.flat[i]

    def test_flatable_getitem_error(self, obj):
        with pytest.raises(IndexError):
            obj.flat[len(obj.flat)]


class TestSequentiable(TestClass):
    """
    Tests the Sequentiable interface.
    """

    def test_sequentiable_serializes_to_args(self, cls, args):
        obj = cls(*args)
        assert list(obj) == list(args)


class TestNormedObject(TestClass):
    """
    Abstract tests for normed objects.
    """

    tol = tol()
    norm = None
    unitary_args = ()

    @pytest.fixture
    def unitary(self):
        return self.base_cls(*self.unitary_args)

    def test_unit_object_has_unity_norm(self, unitary):
        assert abs(unitary.norm(self.norm) - 1.0) < self.tol
        assert abs(unitary.norm_sqr(self.norm) - 1.0) < self.tol
        assert unitary.is_unity(self.norm, tol=self.tol)

    def test_doubled_object_is_not_normalized(self, unitary):
        assert abs((2 * unitary).norm() - 2) < self.tol

    def test_unit_object_is_normalized(self, unitary):
        assert abs((unitary.normalize(self.norm) - unitary)
                   .norm(self.norm)) < self.tol

    def test_stretched_object_has_norm_greater_than_one(self, unitary):
        assert (unitary * 1.1).norm(self.norm) > 1

    def test_shrunk_object_has_norm_smaller_than_one(self, unitary):
        assert (unitary * 0.9).norm(self.norm) < 1

    def test_triangular_identity(self, unitary):
        assert_triangular_identity(unitary, unitary, self.norm)
        assert_triangular_identity(unitary, 2 * unitary, self.norm)
        assert_triangular_identity(unitary, 0 * unitary, self.norm)

    def test_null_vector_is_null(self, unitary):
        assert not unitary.is_null()
        assert (unitary * 0).is_null()
