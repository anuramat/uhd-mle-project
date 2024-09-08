import agent_code.rule_based_agent.callbacks as based


def setup(self):
    based.setup(self)


def act(self, game_state: dict) -> str:
    return based.act(self, game_state)
