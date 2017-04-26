import pytest

from smallvectors.core.mutability import Mutable, Immutable
from smallvectors.tests import base


@pytest.fixture
def tol():
    return 1e-5


def abstract_prop(name):
    @property
    def fget(self):
        raise NotImplementedError('please define %r in your test class' % name)

    return fget


class ClassTester(base.ClassTester):
    """
    Base tester for smallvectors classes.
    """

    def test_object_has_no_dict(self, obj):
        cls_list = list(type(obj).mro())[:-1]
        for subclass in cls_list:
            assert '__slots__' in subclass.__dict__, \
                '%s has no slots' % subclass

        with pytest.raises(AttributeError):
            obj.not_a_valid_attribute_name = None

    def test_class_is_type(self, obj):
        assert obj.__class__ == type(obj)

    def test_assert_constructor_respects_subclass(self, cls, obj):
        assert isinstance(obj, cls)


class TestMutability(ClassTester):
    """
    Tests the mutable/immutable interface
    """

    @pytest.fixture
    def immutable_cls(self, cls):
        if issubclass(cls, Mutable):
            return cls.__immutable_class__
        return cls

    @pytest.fixture
    def mutable_cls(self, cls):
        if issubclass(cls, Immutable):
            return cls.__mutable_class__
        return cls

    @pytest.fixture
    def mutable(self, mutable_cls, args, kwargs):
        return mutable_cls(*args, **kwargs)

    @pytest.fixture
    def immutable(self, immutable_cls, args, kwargs):
        return immutable_cls(*args, **kwargs)

    def test_mutable_from_immutable(self, cls):
        if issubclass(cls, Mutable):
            assert cls is cls.__mutable_class__
            assert issubclass(cls.__immutable_class__, Immutable)
        elif issubclass(cls, Immutable):
            assert cls is cls.__immutable_class__
            assert issubclass(cls.__mutable_class__, Mutable)
        else:
            raise RuntimeError('base_cls is neither mutable nor immutable')

    def test_mutable_equality(self, mutable, mutable_cls, args, kwargs):
        new = mutable_cls(*args, **kwargs)
        assert mutable == new

    def test_immutable_equality(self, immutable, immutable_cls, args, kwargs):
        new = immutable_cls(*args, **kwargs)
        assert immutable == new

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


for _attr, _method in vars(TestMutability).items():
    if _attr.startswith('test_') and not hasattr(DisableMutabilityTests, _attr):
        setattr(DisableMutabilityTests, _attr, None)


class TestFlatable(ClassTester):
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


class TestSequentiable(ClassTester):
    """
    Tests the Sequentiable interface.
    """

    def test_sequentiable_serializes_to_args(self, cls, args):
        obj = cls(*args)
        assert list(obj) == list(args)

