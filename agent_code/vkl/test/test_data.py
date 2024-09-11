import agent_code.vkl.typing as T
from agent_code.vkl.data import TranDataset
from torch import load
from torch.utils.data.dataloader import DataLoader

batch_size = 2


class TestDataset:
    def test_basic(self):
        input = load("agent_code/watcher/output/trans_0.pt", weights_only=False)
        dataset = TranDataset(input)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        batch = next(iter(dataloader))
        map, aux, action, reward = batch
        assert list(map.shape) == [batch_size, T.N_CHANNELS, T.FIELD_SIZE, T.FIELD_SIZE]
        assert list(aux.shape) == [batch_size, T.AUX_SIZE]
        assert list(action.shape) == [batch_size]
        assert list(reward.shape) == [batch_size]
