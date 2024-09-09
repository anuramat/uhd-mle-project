import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from agent_code.vkl.consts import ACTIONS
import pytorch_lightning as L


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.LazyConv2d(32, 3, padding=0),  # strictly zero padding on input
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(64, 3),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(128, 3),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(128, 3),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(128, 3),
            nn.ReLU(),
            nn.LazyConv2d(128, 3),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
        )

        self.fc = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(len(ACTIONS)),
        )

    def forward(self, map, bomb):
        x = self.conv(map)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, bomb.reshape(-1, 1)), dim=1)
        x = self.fc(x)

        return F.softmax(x, dim=1)


class LitBasicModel(L.LightningModule):
    def __init__(self, model, total_steps):
        super().__init__()
        self.model = model
        self.total_steps = total_steps

    def training_step(self, batch, batch_idx):
        map, bomb, action = batch
        out = self.model(map, bomb)
        action = action
        loss = F.cross_entropy(out, action)

        self.log("train_loss", loss)
        self.log("lr", self.lr_schedulers().get_last_lr()[0])  # pyright:ignore
        return loss

    def configure_optimizers(self):  # pyright:ignore
        # Adam is for some reason invisible to pyright
        # <https://github.com/pytorch/pytorch/issues/134985>
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)  # pyright: ignore
        scheduler = OneCycleLR(
            optimizer=optimizer, max_lr=3e-4, total_steps=self.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
