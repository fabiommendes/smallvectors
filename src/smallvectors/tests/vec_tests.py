from smallvectors.tests import ArithmeticUnittest, unittest
from smallvectors import Vec, mVec, AnyVec, VecOrPoint


class VecClassTest(unittest.TestCase):

    def test_class_is_type(self):
        u = Vec(1, 2)
        self.assertEqual(u.__class__, type(u))

    def test_subclass(self):
        assert issubclass(Vec, AnyVec)
        assert issubclass(mVec, AnyVec)
        assert issubclass(AnyVec, VecOrPoint)

    def test_unique_subclass(self):
        assert Vec[2, float] is Vec[2, float]
        assert Vec[2, int] is Vec[2, int]

    def test_class_parameters(self):
        vec2 = Vec[2, float]
        assert vec2.shape == (2,)
        assert vec2.size == 2
        assert vec2.dtype == float
        assert vec2.parameters == (2, float)
        assert vec2.__name__ == 'Vec[2, float]'

    def test_correct_type_promotion_on_vec_creation(self):
        assert isinstance(Vec(1, 2), Vec[2, int])
        assert isinstance(Vec(1.0, 2.0), Vec[2, float])
        assert isinstance(Vec(1, 2.0), Vec[2, float])
        assert isinstance(Vec(1.0, 2), Vec[2, float])

    def test_vec_equality(self):
        assert Vec(1, 2) == Vec(1, 2)
        assert Vec(1, 2) == Vec(1.0, 2.0)

    def test_vec_equality_with_tuples_and_lists(self):
        assert Vec(1, 2) == [1, 2]
        assert Vec(1, 2) == (1, 2)
        assert Vec(1, 2) == [1.0, 2.0]
        assert Vec(1, 2) == (1.0, 2.0)

    def test_reverse_vec_equality_with_tuples_and_lists(self):
        assert [1, 2] == Vec(1, 2)
        assert (1, 2) == Vec(1, 2)
        assert [1.0, 2.0] == Vec(1, 2)
        assert (1.0, 2.0) == Vec(1, 2)

    def test_vec_promotion_on_arithmetic_operations(self):
        u = Vec(1, 2)
        v = Vec(0.0, 0.0)
        assert isinstance(u + v, Vec[2, float])
        assert isinstance(u - v, Vec[2, float])
        assert isinstance(u * 1.0, Vec[2, float])
        assert isinstance(u / 1.0, Vec[2, float])


class Vec2IntTest(ArithmeticUnittest):
    obj_type = Vec[2, int]

    def names(self):
        Vec = self.obj_type
        u = Vec(1, 2)
        v = Vec(3, 4)
        a_tuple = (1, 2)
        a_list = [1, 2]
        m = 2
        k = 0.5

        add_uv = (4, 6)
        sub_uv = (-2, -2)
        sub_vu = (2, 2)

        mul_mu = (2, 4)
        mul_ku = (0.5, 1.0)
        div_um = (0.5, 1)
        div_ku = (2, 4)

        del Vec
        return locals()

    def test_invalid_add_scalar(self):
        with self.assertRaises(TypeError):
            self.u + 1

    def test_invalid_sub_scalar(self):
        with self.assertRaises(TypeError):
            self.u - 1

    def test_invalid_mul_tuple(self):
        with self.assertRaises(TypeError):
            self.u * (1, 2)

    def test_invalid_mul_vec(self):
        with self.assertRaises(TypeError):
            self.u * self.u

    def test_invalid_div_tuple(self):
        with self.assertRaises(TypeError):
            self.u / (1, 2)
        with self.assertRaises(TypeError):
            (1, 2) / self.u

    def test_invalid_div_vec(self):
        with self.assertRaises(TypeError):
            self.u / self.u

    def test_invalid_div_scalar(self):
        with self.assertRaises(TypeError):
            1 / self.u

    def test_rotated_is_new(self):
        assert self.u.rotated(1.0) is not self.u
        
    def test_rotated_keeps_norm(self):
        for t in range(20):
            Z1 = self.u.norm()
            Z2 = self.u.rotated(6.28 * t / 20).norm()
            assert abs(Z1 - Z2) < 1e-6, (Z1, Z2)

class Vec2FloatTest(Vec2IntTest):
    obj_type = Vec[2, float]
    


# class Vec2DecimalTest(Vec2IntTest):
#    test_type = Vec[2, Decimal]


if __name__ == '__main__':
    unittest.main()
