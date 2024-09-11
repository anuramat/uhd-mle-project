from torch.utils.data import Dataset
from agent_code.vkl.preprocessing import get_map
import agent_code.vkl.typing as T
from agent_code.vkl.reward import get_reward


def pack_tran(old: dict, new: dict, action_string: str, events: list[str]):
    s = T.parse_state(old)
    action = T.action_s2i(action_string)
    map = get_map(s)
    reward = get_reward(
        s,
        new,
        events,
    )
    return T.Transition(map, s.self.has_bomb, action, reward)


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
