class TestMat3x3(SquareBase):
    base_shape = (3, 3)
    base_args = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    base_args__other = [1, 2, 3], [1, 2, 3], [1, 2, 3]
    base_args__zero = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    base_args__add_ab = [2, 4, 6], [5, 7, 9], [8, 10, 12]
    base_args__sub_ab = [0, 0, 0], [3, 3, 3], [6, 6, 6]
    base_args__I = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    base_args__smul = [2, 4, 6], [8, 10, 12], [14, 16, 18]
    base_args__mul = [6, 12, 18], [15, 30, 45], [24, 48, 72]


