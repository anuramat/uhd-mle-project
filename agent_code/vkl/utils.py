import torch
from torch import float32, zeros_like, tensor


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

    n = field.shape[1]
    x, y = state["self"][3]
    x_coord = torch.arange(-x, n - x).repeat(n, 1)
    y_coord = torch.arange(-y, n - y).repeat(n, 1).transpose(0, 1)

    for bomb in state["bombs"]:
        pos, ticks = bomb
        bombs[*pos] = ticks

    coins_list = state["coins"]
    for coin in coins_list:
        x, y = coin
        coins[x, y] = 1

    # -1 for bombless players,
    # +1 for players with a bomb
    for player in state["others"]:
        has_bomb = player[2]
        x, y = player[3]
        players[x, y] = -1
        if has_bomb:
            players[x, y] = 1

    return torch.stack(
        [
            field,
            explosion_map,
            bombs,
            coins,
            players,
            x_coord,
            y_coord,
        ]
    ).to(dtype=float32)
