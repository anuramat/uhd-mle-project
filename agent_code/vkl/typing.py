from typing import NamedTuple, NewType
from numpy.typing import ArrayLike
from torch import Tensor

N_CHANNELS = 8
AUX_SIZE = 3
FIELD_SIZE = 17

BOMB = "BOMB"
UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
WAIT = "WAIT"

ACTIONS: list[str] = [
    BOMB,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    WAIT,
]

CRATE = 1
WALL = -1
FREE = 0

Position = NamedTuple("Position", [("x", int), ("y", int)])
Player = NamedTuple(
    "Player",
    [
        ("name", str),
        ("score", int),
        ("has_bomb", bool),
        ("pos", Position),
    ],
)
Bomb = NamedTuple(
    "Bomb",
    [
        (
            "pos",
            Position,
        ),
        (
            "ticks_left",
            int,
        ),
    ],
)
State = NamedTuple(
    "State",
    [
        ("round", int),
        ("step", int),
        ("field", ArrayLike),
        ("bombs", list[Bomb]),
        ("explosion_map", ArrayLike),
        ("coins", list[Position]),
        ("self", Player),
        ("others", list[Player]),
        ("user_input", str | None),
    ],
)


Action = NewType("Action", int)

Aux = NamedTuple(
    "Aux",
    [
        ("has_bomb", bool),
        ("score", int),
        ("step", int),
    ],
)

Transition = NamedTuple(
    "Transition",
    [
        (
            "map",
            Tensor,
        ),
        (
            "aux",
            Aux | Tensor,
        ),
        (
            "action",
            Action,
        ),
        (
            "reward",
            int | float,
        ),
    ],
)


def __parse_state(state_dict: dict) -> State:
    state_dict["self"] = Player(*state_dict["self"])
    state_dict["others"] = [Player(*i) for i in state_dict["others"]]
    state_dict["coins"] = [Position(*i) for i in state_dict["coins"]]
    state_dict["bombs"] = [Bomb(*i) for i in state_dict["bombs"]]
    return State(**state_dict)


def parse_state(s: dict | State) -> State:
    if type(s) is dict:
        s = __parse_state(s)
    elif type(s) is not State:
        raise TypeError
    return s


def action_i2s(i: Action) -> str:
    return ACTIONS[i]


__action_str2int_dict: dict[str, int] = {}
for i, v in enumerate(ACTIONS):
    __action_str2int_dict[v] = i


def action_s2i(s: str) -> Action:
    return Action(__action_str2int_dict[s])
