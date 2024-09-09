import torch
import torch.nn as nn
import torch.nn.functional as F
from agent_code.vkl.consts import N_CHANNELS, ACTIONS


class BasicModel(nn.Module):
    def __init__(self, radius):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        offset = 4
        self.conv_output_dim = 32 * (radius - offset) ** 2

        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, bombful, map):
        x = self.conv(map)

        x = x.reshape(-1, self.conv_output_dim)
        x = torch.cat((x, bombful.reshape(-1, 1)), dim=1)

        x = self.fc(x)

        return F.softmax(x, dim=1)
