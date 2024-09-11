from torch import load, tensor, multinomial
import agent_code.vkl.typing as T
from agent_code.vkl.preprocessing import get_map


def setup(self):
    self.model = load("output/model.pt").eval()


def act(self, game_state: dict):
    map = get_map(game_state).unsqueeze(0)
    player = game_state["self"]
    bomb = player[2]

    proba = self.model(map.float(), tensor(bomb).float().unsqueeze(0)).flatten()

    if not bomb:
        proba[T.action_s2i(T.BOMB)] = 0

    idx = T.Action(int(multinomial(proba, num_samples=1)))
    action = T.action_i2s(idx)

    return action
