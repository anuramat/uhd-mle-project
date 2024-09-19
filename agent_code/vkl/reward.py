import events as e
from collections import defaultdict
import agent_code.vkl.typing as T


def rew2ret(trans: list[T.Transition], discount=0.9) -> list[T.Transition]:
    returns = [t.reward for t in trans]
    for i in reversed(range(len(returns) - 1)):
        returns[i] += discount * returns[i + 1]

    ret_trans = [
        T.Transition(map=t.map, aux=t.aux, action=t.action, reward=returns[i])
        for i, t in enumerate(trans)
    ]

    return ret_trans


__table = defaultdict(
    int,
    {
        e.INVALID_ACTION: -10,
        e.CRATE_DESTROYED: 1,
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 20,
        e.GOT_KILLED: -100,
        e.KILLED_SELF: -200,
        # idk
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0,
    },
)


def get_reward(old: dict | T.State, new: dict | T.State, events: list[str]) -> int:
    old = T.parse_state(old)
    new = T.parse_state(new)

    total = 0

    for event, reward in __table.items():
        if event in events:
            total += reward

    return total
