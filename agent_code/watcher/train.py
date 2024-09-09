from agent_code.vkl.utils import pack_move
from torch import save

copy_counter = 0


def setup_training(self):
    self.moves = []
    self.games_played = 0  # for checkpoints, unused for now
    global copy_counter
    self.number = copy_counter
    copy_counter += 1


def game_events_occurred(self, state, action_string: str, new_state, events):
    move = pack_move(state, action_string)
    self.moves.append(move)


def end_of_round(self, state, action, events):
    self.games_played += 1
    if self.games_played % 10 == 0:
        save(self.moves, f"data/moves_{self.number}.pt")
