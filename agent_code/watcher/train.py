from agent_code.vkl.utils import get_map
from collections import namedtuple
from torch import save

Move = namedtuple("Move", ["map", "player", "action"])


def setup_training(self):
    self.moves = []
    self.rounds_played = 0


def game_events_occurred(self, state, action, new_state, events):
    move = Move(map=get_map(state), player=state["self"], action=action)
    # TODO save action as a number, save only necessary fields of self
    self.moves.append(move)


def end_of_round(self, state, action, events):
    self.rounds_played += 1
    save(self.moves, f"moves_{self.rounds_played}.pt")
    self.moves = []
