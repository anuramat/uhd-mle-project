import torch
from torch import tensor
from agent_code.vkl.utils import get_pov


map_base = torch.ones(1, 3, 3)
map_random = torch.rand(1, 5, 5)


class TestGetPov:
    def test_random_center(self):

        random = get_pov(map_random, (2, 2))
        random_expected = map_random[:, 1:4, 1:4]
        print(random)
        print(random_expected)
        assert torch.equal(random, random_expected)

    def test_trivial(self):
        trivial = get_pov(map_random, (2, 2), 2)
        assert torch.equal(trivial, map_random)

    def test_corner(self):
        corner = get_pov(map_base, (0, 0))
        print(corner)
        assert torch.equal(corner, map_base)  # TODO
