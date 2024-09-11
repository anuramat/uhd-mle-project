from torch import load, tensor, Tensor
import agent_code.vkl.typing as T
from agent_code.vkl.preprocessing import get_map


def setup(self):
    self.model = load("output/model.pt").eval()
    self.training = False


def act(self, s: dict | T.State):
    s = T.parse_state(s)
    map = get_map(s).unsqueeze(0)

    proba = self.model(
        map.float(), tensor(s.self.has_bomb).float().unsqueeze(0)
    ).flatten()

    if not self.training:
        proba = filter_proba(proba, s, map)

    p = {T.ACTIONS[i]: round(float(proba[i]), 5) for i in range(len(T.ACTIONS))}

    if not self.training:
        print(p)

    action = T.action_i2s(int(proba.argmax()))

    return action


# remove illegal moves
def filter_proba(proba: Tensor, s: T.State, map: Tensor) -> Tensor:
    if not s.self.has_bomb:
        proba[T.action_s2i(T.BOMB)] = float("-inf")
    x, y = s.self.pos
    if illegal_cell(map, x - 1, y):
        proba[T.action_s2i(T.LEFT)] = float("-inf")
    if illegal_cell(map, x + 1, y):
        proba[T.action_s2i(T.RIGHT)] = float("-inf")
    if illegal_cell(map, x, y - 1):
        proba[T.action_s2i(T.UP)] = float("-inf")
    if illegal_cell(map, x, y + 1):
        proba[T.action_s2i(T.DOWN)] = float("-inf")
    return proba


def illegal_cell(map: Tensor, x: int, y: int) -> bool:
    return round(float(map[0, 0, x, y])) != T.FREE or round(float(map[0, 2, x, y])) != 0
