from torch import tensor, randn
from agent_code.vkl.models import BasicModel
from agent_code.vkl.utils import ACTIONS


class TestModels:
    def basic(self):
        radius = 9
        n_channels = 5

        model = BasicModel(n_channels, 9)

        bombful = tensor([1])
        map = randn(1, n_channels, radius, radius)

        output = model(bombful, map)
        assert list(output.shape) == [1, len(ACTIONS)]
