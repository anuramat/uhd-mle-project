from agent_code.vkl.data import pack_tran
from agent_code.vkl.reward import rew2ret
from torch import save
from os import environ
import agent_code.vkl.typing as T
import events as e

__copy_counter = 0


def setup_training(self):
    # HACK we want to save data from multiple copies of an agent
    global __copy_counter
    self.agent_id = __copy_counter
    __copy_counter += 1

    # HACK so that we know when to do the final checkpoint
    self.n_games = int(environ["N_GAMES"])

    self.games_played = 0  # for checkpointing

    # buffer for the current game
    self.trans = []
    # calculate Q before dumping to output buffer

    self.output = []  # total buffer


def game_events_occurred(
    self, state_dict: dict, action_string: str, new_state: dict, events: list[str]
):
    tran = pack_tran(state_dict, new_state, action_string, events)
    self.trans.append(tran)


def end_of_round(self, state, action, events):
    self.output += rew2ret(self.trans)
    self.trans = []
    self.games_played += 1
    if self.games_played == self.n_games:
        save(self.output, f"output/trans_{self.agent_id}.pt")
