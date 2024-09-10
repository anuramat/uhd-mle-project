from torch import load, tensor, multinomial
from agent_code.vkl.consts import INT2STR, BOMB, STR2INT
from agent_code.vkl.utils import get_map


def setup(self):
    self.model = load("output/model.pt").eval()


def act(self, game_state: dict):
    map = get_map(game_state).unsqueeze(0)
    player = game_state["self"]
    bomb = player[2]

    proba = self.model(map.float(), tensor(bomb).float().unsqueeze(0)).flatten()

    if not bomb:
        proba[STR2INT[BOMB]] = 0

    idx = int(multinomial(proba, num_samples=1))
    action = INT2STR[idx]

    return action
