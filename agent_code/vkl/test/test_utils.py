from torch import tensor, rand, equal, ones
from agent_code.vkl.utils import get_pov


class TestGetPov:
    def test_trivial(self):
        input = rand(2, 5, 5)
        result = get_pov(input, (2, 2), 2)
        expected = input
        assert equal(result, expected)

    def test_random_center(self):
        input = rand(2, 5, 5)
        result = get_pov(input, (2, 2))
        expected = input[..., 1:4, 1:4]
        assert equal(result, expected)

    def test_small_corner(self):
        input = ones(1, 3, 3)
        result = get_pov(input, (2, 0))
        expected = tensor(
            [
                [
                    [-2.0, 1.0, 1.0],
                    [-2.0, 1.0, 1.0],
                    [-2.0, -2.0, -2.0],
                ]
            ]
        )
        assert equal(result, expected)

    def test_big_corner(self):
        input = ones(1, 3, 3)
        result = get_pov(input, (0, 0), 3)
        expected = tensor(
            [
                [
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, 1.0, 1.0, 1.0, -2.0],
                    [-2.0, -2.0, -2.0, 1.0, 1.0, 1.0, -2.0],
                    [-2.0, -2.0, -2.0, 1.0, 1.0, 1.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                ]
            ]
        )
        assert equal(result, expected)

    def test_uneven(self):
        input = ones(1, 2, 30)
        result = get_pov(input, (0, 0), 3)
        expected = tensor(
            [
                [
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
                    [-2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                ]
            ]
        )
        assert equal(result, expected)
