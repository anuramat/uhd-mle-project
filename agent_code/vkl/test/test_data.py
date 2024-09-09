import torch
from agent_code.vkl.consts import N_CHANNELS
from agent_code.vkl.data import MoveDataset
from torch import load
from torch.utils.data.dataloader import DataLoader

batch_size = 123


class TestDataset:
    def test_basic(self):
        input = load("agent_code/watcher/output/moves_0.pt", weights_only=False)
        dataset = MoveDataset(input)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        map, bomb, action = next(iter(dataloader))
        assert list(map.shape) == [batch_size, N_CHANNELS, 3, 3]
        assert list(bomb.shape) == [batch_size]
        print(bomb.dtype)
        assert bomb.dtype == torch.bool
        assert list(action.shape) == [batch_size]
