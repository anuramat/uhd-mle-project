from torch import float32, load, tensor, Tensor
import agent_code.vkl.typing as T
from agent_code.vkl.preprocessing import get_aux, get_map
from os.path import join
from os import environ
from random import choice, random


def setup(self):
    path = join(environ["PWD"], environ["MODEL"])
    self.model = load(path, weights_only=False).eval()
    self.training = False


def act(self, s: dict | T.State):
    s = T.parse_state(s)
    map = get_map(s).unsqueeze(0)

    if self.training:
        if random() < self.epsilon:
            return random_move(s, map)

    aux = tensor(get_aux(s), dtype=float32).unsqueeze(
        0
    )  # TODO maybe move torch() to get_aux

    # do the heavy lifting
    q = self.model(map, aux).flatten()

    # some stats
    if not self.training:
        pretty_q = {T.ACTIONS[i]: round(float(q[i]), 5) for i in range(len(T.ACTIONS))}
        print(pretty_q)

    # remove illegal moves
    q = filter_proba(q, s, map)

    action = T.action_i2s(int(q.argmax()))

    return action


def random_move(s: T.State, map: Tensor) -> str:
    proba = tensor([1] * len(T.ACTIONS))
    proba = filter_proba(proba, s, map, replacement=0)
    allowed_actions = [T.action_i2s(int(i)) for i in proba.nonzero().flatten()]
    action = choice(allowed_actions)
    return action


# remove illegal moves
def filter_proba(
    proba: Tensor, s: T.State, map: Tensor, replacement=float("-inf")
) -> Tensor:
    if not s.self.has_bomb:
        proba[T.action_s2i(T.BOMB)] = replacement
    x, y = s.self.pos
    if illegal_cell(map, x - 1, y):
        proba[T.action_s2i(T.LEFT)] = replacement
    if illegal_cell(map, x + 1, y):
        proba[T.action_s2i(T.RIGHT)] = replacement
    if illegal_cell(map, x, y - 1):
        proba[T.action_s2i(T.UP)] = replacement
    if illegal_cell(map, x, y + 1):
        proba[T.action_s2i(T.DOWN)] = replacement
    return proba


def illegal_cell(map: Tensor, x: int, y: int) -> bool:
    return (
        round(float(map[0, 0, x, y])) != T.FREE
        or round(float(map[0, 2, x, y])) != 0  # bomb
        or round(float(map[0, 4, x, y])) != 0  # player
    )
