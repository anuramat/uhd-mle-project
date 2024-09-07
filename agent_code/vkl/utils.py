import torch

from torch import tensor, zeros_like, Tensor

from torch.nn.functional import pad

OOB_MAP_VALUE = -2


def get_map(state):
    """
    returns the state of the game encoded in a 5*h*w tensor
    omits the players has_bomb property
    """
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
