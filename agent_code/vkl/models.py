import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from agent_code.vkl.consts import N_CHANNELS, ACTIONS
import pytorch_lightning as L


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
        self.conv_output_dim = 32 * (radius * 2 + 1 - offset) ** 2

        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, map, bombful):
        x = self.conv(map)

        x = x.reshape(-1, self.conv_output_dim)
        x = torch.cat((x, bombful.reshape(-1, 1)), dim=1)

        x = self.fc(x)

        return F.softmax(x, dim=1)


class LitBasicModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        map, bomb, action = batch
        out = self.model(map, bomb)
        action = action.float()
        loss = F.cross_entropy(out, action)
        self.log("train_loss", loss)

    def configure_optimizers(self):
        # Adam is for some reason invisible to pyright
        # <https://github.com/pytorch/pytorch/issues/134985>
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)  # pyright: ignore
        return optimizer
