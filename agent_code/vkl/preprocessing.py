import torch
from torch import float32, zeros_like, tensor, Tensor
import agent_code.vkl.typing as T


def get_map(s: T.State | dict) -> Tensor:
    """
    returns the state of the game encoded in a 5*h*w tensor
    omits the players properties
    """

    s = T.parse_state(s)

    field = tensor(s.field)
    explosion_map = tensor(s.explosion_map)
    bombs = zeros_like(field)
    coins = zeros_like(field)
    player_bomb = zeros_like(field)
    player_score = zeros_like(field)

    n = field.shape[1]
    x, y = s.self.pos
    x_coord = torch.arange(-x, n - x).repeat(n, 1).t()
    y_coord = torch.arange(-y, n - y).repeat(n, 1)

    for pos, ticks in s.bombs:
        bombs[*pos] = ticks + 1  # damn, I just got bamboozled

    for x, y in s.coins:
        coins[x, y] = 1

    for p in s.others:
        player_bomb[*p.pos] = int(p.has_bomb) * 2 - 1
        player_score[*p.pos] = p.score

    return torch.stack(
        [
            field,
            explosion_map,
            bombs,
            coins,
            player_bomb,
            player_score,  # I think this is always zero lmao
            x_coord,
            y_coord,
        ]
    ).to(dtype=float32)


def get_aux(s: T.State | dict) -> T.Aux:
    s = T.parse_state(s)
    return T.Aux(has_bomb=s.self.has_bomb, score=s.self.score, step=s.step)
