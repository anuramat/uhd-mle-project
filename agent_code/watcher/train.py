from agent_code.vkl.utils import get_map
from collections import namedtuple
from torch import save

Move = namedtuple("Move", ["map", "player", "action"])


class Saver:
    def __init__(self, path):
        self.moves = []
        self.path = path

    def __del__(self):
        save(self.moves, self.path)


def setup_training(self):
    self.saver = Saver("moves.pt")


def game_events_occurred(self, state, action, new_state, events):
    move = (get_map(state), state["self"], action)
    # TODO save action as a number, save only necessary fields of self
    self.saver.moves.append(move)


def end_of_round(self, state, action, events):
    pass
