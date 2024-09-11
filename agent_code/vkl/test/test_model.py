from torch import tensor, randn
from agent_code.vkl.consts import FIELD_SIZE, N_CHANNELS, ACTIONS
from agent_code.vkl.models import MyBelovedCNN


class TestModels:
    def test_basic(self):
        model = MyBelovedCNN().eval()

        bomb = tensor([1])
        map = randn(1, N_CHANNELS, FIELD_SIZE, FIELD_SIZE)

        output = model(map, bomb)
        assert list(output.shape) == [1, len(ACTIONS)]
