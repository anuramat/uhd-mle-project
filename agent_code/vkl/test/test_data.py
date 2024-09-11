import agent_code.vkl.typing as T
from agent_code.vkl.data import MoveDataset
from torch import load
from torch.utils.data.dataloader import DataLoader

batch_size = 2


class TestDataset:
    def test_basic(self):
        input = load("agent_code/watcher/output/trans_0.pt", weights_only=False)
        dataset = MoveDataset(input)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        map, bomb, action = next(iter(dataloader))
        assert list(map.shape) == [batch_size, T.N_CHANNELS, T.FIELD_SIZE, T.FIELD_SIZE]
        assert list(bomb.shape) == [batch_size]
        assert list(action.shape) == [batch_size]
