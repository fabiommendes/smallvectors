import pytest
import smallvectors as sv


class ConvFunctionTester:
    function = None
    base_cls = None
    base_args = (1, 2)

    @pytest.fixture
    def conv(self):
        return type(self).function

    @pytest.fixture
    def obj(self, cls, args):
        return cls(*args)

    @pytest.fixture
    def cls(self):
        return self.base_cls

    @pytest.fixture
    def args(self):
        return self.base_args

    def test_convertion_to_object(self, conv, obj):
        assert obj is conv(obj)

    def test_convertion_equivalence(self, conv, obj, args):
        assert obj == conv(args)


class TestAsVector(ConvFunctionTester):
    function = sv.asvector
    base_cls = sv.Vec


class TestAsDirection(ConvFunctionTester):
    function = sv.asdirection
    base_cls = sv.Direction


class TestAsPoint(ConvFunctionTester):
    function = sv.aspoint
    base_cls = sv.Point


class TestAsMatrix(ConvFunctionTester):
    function = sv.asmatrix
    base_cls = sv.Mat
    base_args = [(1, 2), (3, 4)]