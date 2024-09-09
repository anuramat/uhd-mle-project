from torch import load, argmax, tensor
from agent_code.vkl.consts import INT2STR, BOMB, STR2INT
from agent_code.vkl.utils import get_map


def setup(self):
    self.model = load("output/model.pt").eval()


def act(self, game_state: dict):
    map = get_map(game_state).unsqueeze(0)
    player = game_state["self"]
    bomb = player[2]

    proba = self.model(map.float(), tensor(bomb).float().unsqueeze(0))
    if not bomb:
        proba[STR2INT[BOMB]] = 0
        print("BOMB REJECTED")
    index = int(argmax(proba))
    action = INT2STR[index]
    print(proba)

    return action
