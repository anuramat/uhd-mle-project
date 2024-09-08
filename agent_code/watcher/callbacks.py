import agent_code.rule_based_agent.callbacks as based
from agent_code.vkl.utils import WAIT, ACTIONS


def setup(self):
    based.setup(self)


def act(self, game_state: dict) -> str:
    action = based.act(self, game_state)
    if action not in ACTIONS:
        return WAIT
    return action
