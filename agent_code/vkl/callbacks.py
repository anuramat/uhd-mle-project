from torch import load, argmax, tensor
from agent_code.vkl.consts import ACTIONS, INT2STR
from agent_code.vkl.utils import get_map, get_pov
from random import random, choice


def setup(self):
    self.model, self.fov = load("output/model.pt")


def act(self, game_state: dict):
    eps = 0.1
    map = get_map(game_state)

    player = game_state["self"]
    bomb = player[2]
    x, y = player[3]
    pov = get_pov(map, (y, x), self.fov)

    proba = self.model(pov.float(), tensor(bomb).float())
    print(proba)
    index = int(argmax(proba))
    action = INT2STR[index]
    if random() < eps:
        action = choice(ACTIONS)

    return action
