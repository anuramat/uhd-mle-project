from agent_code.vkl.data import pack_tran
from agent_code.vkl.reward import rew2ret
from torch import save
from os import environ

__copy_counter = 0
__time_to_save = False


def setup_training(self):
    self.training = True
    self.epsilon = 0.1
    self.quit_on_next_episode = False
    # we want to save data from multiple copies of an agent
    global __copy_counter
    self.agent_id = __copy_counter
    __copy_counter += 1

    # so that we know when to do the final checkpoint
    self.n_trans = int(environ["N_TRANS"])

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

    if self.quit_on_next_episode:
        raise KeyboardInterrupt("hehe")

    global __time_to_save
    if len(self.output) > self.n_trans:
        __time_to_save = True
    if __time_to_save:
        save(self.output, f"output/{self.source_name}_{self.agent_id}.pt")
        self.quit_on_next_episode = True
