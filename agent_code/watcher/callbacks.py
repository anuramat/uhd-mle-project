import agent_code.rule_based_agent.callbacks as based
import agent_code.vkl.callbacks as vkl
import agent_code.vkl.typing as T
from os import environ

model = vkl
model_name = environ["MODEL"]
if model_name == "rule_based_agent":
    model = based


def setup(self):
    model.setup(self)
    self.source_name = model_name
    if self.shadow:
        self.source_name += "_shadow"
    self.scenario_name = environ["SCENARIO_NAME"]


def act(self, game_state: dict) -> str:
    action = model.act(self, game_state)
    if action not in T.ACTIONS:
        return T.WAIT
    return action
