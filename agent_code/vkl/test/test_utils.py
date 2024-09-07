import torch
from agent_code.vkl.utils import get_pov


def test_get_pov():
    base = torch.ones(1, 3, 3)
    print(base)

    trivial = get_pov(base, (1, 1), 1)
    print(trivial)

    assert torch.equal(base, trivial)
