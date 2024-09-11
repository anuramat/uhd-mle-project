from torch.utils.data import Dataset
from agent_code.vkl.preprocessing import get_aux, get_map
import agent_code.vkl.typing as T
from agent_code.vkl.reward import get_reward
from torch import tensor, float32


def pack_tran(old: dict, new: dict, action_string: str, events: list[str]):
    s = T.parse_state(old)
    return T.Transition(
        get_map(s),
        get_aux(s),
        T.action_s2i(action_string),
        get_reward(
            s,
            new,
            events,
        ),
    )


class TranDataset(Dataset):
    def __init__(
        self,
        data: list[T.Transition],
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t: T.Transition = self.data[idx]
        return T.Transition(
            map=t.map,
            aux=tensor(t.aux, dtype=float32),
            action=t.action,
            reward=tensor(t.reward, dtype=float32),
        )
