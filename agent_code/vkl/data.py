from torch.utils.data import Dataset
from agent_code.vkl.utils import get_map
from agent_code.vkl.consts import *


def pack_move(state: dict, action_string: str):
    player = state["self"]
    bomb = player[2]
    action_number = STR2INT[action_string]

    return (get_map(state), bomb, action_number)


class MoveDataset(Dataset):
    def __init__(
        self,
        data,
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
