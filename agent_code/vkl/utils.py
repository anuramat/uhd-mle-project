from typing import NamedTuple
import torch
from torch import float32, zeros_like, tensor

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


def get_map(state):
    """
    returns the state of the game encoded in a 5*h*w tensor
    omits the players properties
    """
    field = tensor(state["field"])
    explosion_map = tensor(state["explosion_map"])
    bombs = zeros_like(field)
    coins = zeros_like(field)
    player_bomb = zeros_like(field)
    player_score = zeros_like(field)

    n = field.shape[1]
    x, y = state["self"][3]
    x_coord = torch.arange(-x, n - x).repeat(n, 1)
    y_coord = torch.arange(-y, n - y).repeat(n, 1).transpose(0, 1)

    for bomb in state["bombs"]:
        pos, ticks = bomb
        bombs[*pos] = ticks + 1  # damn, I just got bamboozled

    coins_list = state["coins"]
    for coin in coins_list:
        x, y = coin
        coins[x, y] = 1

    for player in state["others"]:
        player = Player(*player)
        x, y = player.pos
        player_bomb[x, y] = int(player.has_bomb) * 2 - 1
        player_score[x, y] = player.score

    return torch.stack(
        [
            field,
            explosion_map,
            bombs,
            coins,
            player_bomb,
            player_score,
            x_coord,
            y_coord,
        ]
    ).to(dtype=float32)
