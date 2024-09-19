import agent_code.rule_based_agent.callbacks as based
import agent_code.vkl.callbacks as vkl
import agent_code.vkl.typing as T
import os

model = based
model_name = os.environ["MODEL"]
if model_name == "vkl":
    model = vkl
elif model_name == "rule_based_agent":
    model = based
else:
    raise ValueError(f"Illegal $MODEL={model_name} set in the ./datagen.sh script")


def setup(self):
    model.setup(self)
    self.source_name = model_name


def act(self, game_state: dict) -> str:
    action = model.act(self, game_state)
    if action not in T.ACTIONS:
        return T.WAIT
    return action
