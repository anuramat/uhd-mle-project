from torch import tensor, randn
import agent_code.vkl.typing as T
from agent_code.vkl.models import MyBelovedCNN


class TestModels:
    def test_basic(self):
        model = MyBelovedCNN().eval()

        bomb = tensor([1])
        map = randn(1, T.N_CHANNELS, T.FIELD_SIZE, T.FIELD_SIZE)

        output = model(map, bomb)
        assert list(output.shape) == [1, len(T.ACTIONS)]
