import torch
from torch import tensor
from agent_code.vkl.utils import get_pov


map_base = torch.ones(1, 3, 3)
map_random = torch.rand(1, 5, 5)


class TestGetPov:
    def test_random_center(self):
        result = get_pov(map_random, (2, 2))
        expected = map_random[..., 1:4, 1:4]
        assert torch.equal(result, expected)

    def test_trivial(self):
        result = get_pov(map_random, (2, 2), 2)
        expected = map_random
        assert torch.equal(result, expected)

    def test_small_corner(self):
        result = get_pov(map_base, (0, 0))
        expected = tensor(
            [
                [
                    [-2.0, -2.0, -2.0],
                    [-2.0, 1.0, 1.0],
                    [-2.0, 1.0, 1.0],
                ]
            ]
        )
        assert torch.equal(result, expected)

    def test_big_corner(self):
        result = get_pov(map_base, (0, 0), 3)
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
        print(result)
        assert torch.equal(result, expected)
