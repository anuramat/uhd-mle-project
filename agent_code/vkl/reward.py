import events as e
import agent_code.vkl.typing as T


def get_reward(old: dict | T.State, new: dict | T.State, events: list[str]) -> T.Reward:
    old = T.parse_state(old)
    new = T.parse_state(new)
    return T.Reward(0)


def rew2ret(trans: list[T.Transition]) -> list[T.Transition]:
    return trans
