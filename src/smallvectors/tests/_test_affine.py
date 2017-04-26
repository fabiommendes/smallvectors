import pytest
from smallvectors import Affine
from smallvectors.tests import abstract as base

# class AffineBase(base.TestMutability):
#     mutable_cls = mAffine
#     immutable_cls = Affine
#
# class TestAffine2d(AffineBase):
#     sdfs


def test_affine_shape():
    T = Affine[2, float]
    assert T.size == 6


