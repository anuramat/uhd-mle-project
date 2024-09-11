from torch import randn
import agent_code.vkl.typing as T
from agent_code.vkl.models import MyBelovedCNN


class TestModels:
    def test_basic(self):
        model = MyBelovedCNN().eval()

        aux = randn(1, T.AUX_SIZE)
        map = randn(1, T.N_CHANNELS, T.FIELD_SIZE, T.FIELD_SIZE)

        output = model(map, aux)
        assert list(output.shape) == [1, len(T.ACTIONS)]
