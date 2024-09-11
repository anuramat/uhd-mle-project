import events as e
import agent_code.vkl.typing as T


def rew2ret(trans: list[T.Transition], discount=0.9) -> list[T.Transition]:
    returns = [t.reward for t in trans]
    for i in reversed(range(len(returns) - 1)):
        returns[i] += discount * returns[i + 1]  # pyright:ignore

    return trans


def get_reward(old: dict | T.State, new: dict | T.State, events: list[str]) -> int:
    old = T.parse_state(old)
    new = T.parse_state(new)
    return 0
