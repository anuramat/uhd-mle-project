from torch import tensor, Tensor
from torch.utils.data import Dataset
from agent_code.vkl.utils import get_map, get_pov
from agent_code.vkl.consts import *


def pack_move(state: dict, action_string: str):
    player = state["self"]
    bombful = player[2]
    pos = player[3]
    action_number = STR2INT[action_string]

    return (get_map(state), bombful, pos, action_number)


class MoveDataset(Dataset):
    def __init__(
        self, packed: list[tuple[Tensor, bool, tuple[int, int], int]], radius=1
    ):

        self.data = []
        for move in packed:
            x, y = move[2]
            pov = get_pov(move[0], (y, x), radius).float()
            action_list = [0] * len(ACTIONS)
            action_list[move[-1]] = 1
            action = tensor(action_list)
            bomb = tensor(move[1])

            self.data.append((pov, bomb, action))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
