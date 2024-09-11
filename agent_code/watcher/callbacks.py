import agent_code.rule_based_agent.callbacks as based
import agent_code.vkl.typing as T


def setup(self):
    based.setup(self)


def act(self, game_state: dict) -> str:
    action = based.act(self, game_state)
    if action not in T.ACTIONS:
        return T.WAIT
    return action
