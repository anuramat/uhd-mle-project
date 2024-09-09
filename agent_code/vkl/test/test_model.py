from torch import tensor, randn
from agent_code.vkl.consts import N_CHANNELS, ACTIONS
from agent_code.vkl.models import BasicModel


class TestModels:
    def test_basic(self):
        radius = 9
        side = radius * 2 + 1

        model = BasicModel(radius)

        bombful = tensor([1])
        map = randn(1, N_CHANNELS, side, side)

        output = model(map, bombful)
        assert list(output.shape) == [1, len(ACTIONS)]
