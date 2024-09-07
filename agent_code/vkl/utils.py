import torch

from torch import tensor, zeros_like, Tensor

from torch.nn.functional import pad

OOB_MAP_VALUE = -2


def get_map(state):
    bombs = tensor(state["bombs"])
    explosion_maps = tensor(state["explosion_maps"])

    field = tensor(state["field"])
    # -1 = wall, 0 = free, +1 = crate
    # we additionally define -2 for OOB squares

    coins = zeros_like(bombs)
    coins_list = state["coins"]
    for coin in coins_list:
        x, y = coin
        coins[x, y] = 1

    players = zeros_like(bombs)
    # -1 for bombless players, +1 for bombful
    for player in state["others"]:
        # name = player[0] # completely useless, unless we can trashtalk (?)
        # score = player[1]  # XXX maybe avoid people with high scores lmao
        has_bomb = player[2]
        x, y = player[3]
        players[x, y] = -1
        if has_bomb:
            players[x, y] = 1

    return torch.stack([bombs, explosion_maps, field, coins, players])


def get_pov(
    map: Tensor,
    center: tuple[int, int],
    radius: int = 1,  # radius=1 corresponds to 3x3 input
) -> Tensor:

    # NOTE y = 0 is top

    x, y = center

    top = y - radius
    bottom = y + radius

    left = x - radius
    right = x + radius

    h, w = map.shape[-2:]
    right = left + 2 * radius + 1
    bottom = top + 2 * radius + 1

    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [
            max(-left + min(0, right), 0),
            max(-top + min(0, bottom), 0),
            max(right - max(w, left), 0),
            max(bottom - max(h, top), 0),
        ]
        return pad(
            input=map[..., max(top, 0) : bottom, max(left, 0) : right],
            pad=padding_ltrb,
            value=OOB_MAP_VALUE,
        )
    return map[..., top:bottom, left:right]
