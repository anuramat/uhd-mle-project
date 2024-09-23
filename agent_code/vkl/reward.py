import events as e
from torch import Tensor, tensor, float32
from collections import defaultdict
import agent_code.vkl.typing as T


def rew2ret(trans: list[T.Transition], q: list, discount=0.98) -> list[T.Transition]:
    if len(q) == 0:
        return _rew2ret_expert(trans, discount)
    return _rew2ret_dqn(trans, q, discount)


def _rew2ret_expert(trans: list[T.Transition], discount=0.98) -> list[T.Transition]:
    returns = [t.reward for t in trans]
    for i in reversed(range(len(returns) - 1)):
        returns[i] += discount * returns[i + 1]

    ret_trans = [
        T.Transition(map=t.map, aux=t.aux, action=t.action, reward=returns[i])
        for i, t in enumerate(trans)
    ]

    return ret_trans


def _rew2ret_dqn(
    trans: list[T.Transition], q: list, discount=0.9
) -> list[T.Transition]:
    ret_trans = [
        T.Transition(
            map=t.map,
            aux=t.aux,
            action=t.action,
            reward=t.reward + discount * q[i + 1],
        )
        for i, t in enumerate(trans)
    ]

    return ret_trans


__table = defaultdict(
    float,
    {
        e.CRATE_DESTROYED: 1,
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 10,
        e.GOT_KILLED: -20,
        e.KILLED_SELF: -20,
        # in the future might help with very complex behaviour, eg alliancing
        e.OPPONENT_ELIMINATED: 0,
    },
)


def get_reward(old: dict | T.State, new: dict | T.State, events: list[str]) -> Tensor:
    old = T.parse_state(old)
    new = T.parse_state(new)

    total = 0

    for event, reward in __table.items():
        if event in events:
            total += reward

    return tensor(total, dtype=float32)
