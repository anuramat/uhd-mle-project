import agent_code.vkl.typing as T
from agent_code.vkl.data import TranDataset
from torch import load
from torch.utils.data.dataloader import DataLoader
from os import listdir
from os.path import join
from re import search

batch_size = 2


class TestDataset:
    def test_basic(self):
        dir = "agent_code/watcher/data"
        files = [f for f in listdir(dir) if search(r".*\.pt", f)]
        file = files[0]  # take an arbitrary one
        input: list[T.Transition] = load(join(dir, file), weights_only=False)
        dataset = TranDataset(input)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        batch = next(iter(dataloader))
        map, aux, action, reward = batch
        assert list(map.shape) == [batch_size, T.N_CHANNELS, T.FIELD_SIZE, T.FIELD_SIZE]
        assert list(aux.shape) == [batch_size, T.AUX_SIZE]
        assert list(action.shape) == [batch_size]
        assert list(reward.shape) == [batch_size]
