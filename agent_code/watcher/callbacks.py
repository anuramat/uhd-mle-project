import agent_code.rule_based_agent.callbacks as based
import agent_code.vkl.callbacks as vkl
import agent_code.vkl.typing as T
import os

model = based
if os.environ["MODEL"] == "vkl":
    model = vkl


def setup(self):
    model.setup(self)


def act(self, game_state: dict) -> str:
    action = model.act(self, game_state)
    if action not in T.ACTIONS:
        return T.WAIT
    return action
