import torch
from torch import tensor, zeros_like, Tensor
from torch.nn.functional import pad
from agent_code.vkl.consts import *


def get_map(state):
    """
    returns the state of the game encoded in a 5*h*w tensor
    omits the players properties
    """
    # field: -1 = wall, 0 = free, +1 = crate
    # we additionally define -2 for OOB squares
    field = tensor(state["field"])
    explosion_map = tensor(state["explosion_map"])
    bombs = zeros_like(field)
    coins = zeros_like(field)
    players = zeros_like(field)

    for bomb in state["bombs"]:
        pos, ticks = bomb
        bombs[*pos] = ticks

    coins_list = state["coins"]
    for coin in coins_list:
        x, y = coin
        coins[x, y] = 1

    # -1 for bombless players, +1 for bombful
    for player in state["others"]:
        # name = player[0] # completely useless, unless we can trashtalk (?)
        # score = player[1]  # XXX maybe avoid people with high scores lmao
        has_bomb = player[2]
        x, y = player[3]
        players[x, y] = -1
        if has_bomb:
            players[x, y] = 1

    return torch.stack([field, explosion_map, bombs, coins, players]).transpose(
        -1, -2
    )  # c, w, h -> c, h, w


def get_pov(
    map: Tensor,
    center_yx: tuple[int, int],
    radius: int = 1,  # radius=1 corresponds to 3x3 input
) -> Tensor:

    # NOTE y = 0 is top
    # NOTE the field is 17*17 (allegedly), so the max radius is 16+1 for OOB to be included

    y, x = center_yx

    top = y - radius
    bottom = y + radius + 1

    left = x - radius
    right = x + radius + 1

    h, w = map.shape[-2:]

    if not (left < 0 or top < 0 or right > w or bottom > h):
        return map[..., top:bottom, left:right]

    cropped = map[..., max(top, 0) : bottom, max(left, 0) : right]

    padding_lrtb = [
        max(-left + min(0, right), 0),
        max(right - max(w, left), 0),
        max(-top + min(0, bottom), 0),
        max(bottom - max(h, top), 0),
    ]

    return pad(
        input=cropped,
        pad=padding_lrtb,
        mode="constant",
        value=OOB_MAP_VALUE,
    )
