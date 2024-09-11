from torch import load, tensor, multinomial, Tensor
import agent_code.vkl.typing as T
from agent_code.vkl.preprocessing import get_map


def setup(self):
    self.model = load("output/model.pt").eval()


def act(self, s: dict | T.State):
    s = T.parse_state(s)
    map = get_map(s).unsqueeze(0)

    proba = self.model(
        map.float(), tensor(s.self.has_bomb).float().unsqueeze(0)
    ).flatten()
    proba = filter_proba(proba, s, map)

    idx = T.Action(int(multinomial(proba, num_samples=1)))
    action = T.action_i2s(idx)

    p = {T.ACTIONS[i]: round(float(proba[i]), 5) for i in range(len(T.ACTIONS))}
    print(p)

    return action


# remove illegal moves
def filter_proba(proba: Tensor, s: T.State, map: Tensor) -> Tensor:
    if not s.self.has_bomb:
        proba[T.action_s2i(T.BOMB)] = 0
    x, y = s.self.pos
    if illegal_cell(map, x - 1, y):
        proba[T.action_s2i(T.LEFT)] = 0
    if illegal_cell(map, x + 1, y):
        proba[T.action_s2i(T.RIGHT)] = 0
    if illegal_cell(map, x, y - 1):
        proba[T.action_s2i(T.UP)] = 0
    if illegal_cell(map, x, y + 1):
        proba[T.action_s2i(T.DOWN)] = 0
    return proba


def illegal_cell(map: Tensor, x: int, y: int) -> bool:
    return round(float( map[0, 0, x, y])) != T.FREE or round(float(map[0, 2, x, y])) != 0
