from types import SimpleNamespace
from agent_code.vkl.preprocessing import get_aux, get_map
from os import environ
from os.path import join
from random import choice, random
from torch import float32, load, softmax, tensor, Tensor, no_grad, multinomial
import agent_code.rule_based_agent.callbacks as based
import torch
import agent_code.vkl.typing as T


def setup(self):
    # based
    self.shadow = True if environ.get("SHADOW") else False
    self.fake = SimpleNamespace()
    based.setup(self.fake)

    self.training = False
    path = join(environ["PWD"], environ["MODEL"])
    self.device = "cuda" if environ.get("CUDA") else "cpu"
    self.model = load(path, weights_only=False).eval().to(self.device)
    self.q = []


torch.set_printoptions(linewidth=400)


def act(self, state: dict | T.State):
    # prepare shit
    s = T.parse_state(state)
    map = get_map(s)
    if not self.training:
        print("-" * 200)
        for i in range(map.shape[0]):
            print(map[i, :, :])
    map = map.unsqueeze(0)
    aux = tensor(get_aux(s), dtype=float32).unsqueeze(0)

    # do the heavy lifting
    with no_grad():
        q = self.model(map.to(self.device), aux.to(self.device)).flatten().to("cpu")

    # some stats
    if not self.training:
        pretty_q = {T.ACTIONS[i]: round(float(q[i]), 5) for i in range(len(T.ACTIONS))}
        print(pretty_q)

    # remove illegal moves
    q = filter_proba(q, s, map)

    # write Q
    if self.training:
        self.q.append(q.max())

    # epsilon
    if self.training:
        if random() < self.epsilon:
            return random_move(s, map)

    # act
    if self.shadow:
        return based.act(self.fake, state)

    idx = my_argmax(q, self.training)
    action = T.action_i2s(idx)
    print(action)
    return action


def my_argmax(arg: Tensor, soft: bool) -> int:
    if not soft:
        return int(arg.argmax())
    proba = softmax(arg, 0)
    return int(multinomial(proba, num_samples=1))


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
