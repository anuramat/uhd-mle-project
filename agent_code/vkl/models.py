import torch
from torch import Tensor
from torch.nn import (
    ReLU,
    LazyConv2d,
    Sequential,
    LazyBatchNorm2d,
    LazyBatchNorm1d,
    Module,
    ModuleList,
    LazyLinear,
)
from torch.nn.functional import softmax, cross_entropy
from torch.optim.lr_scheduler import OneCycleLR
from agent_code.vkl.consts import ACTIONS
import pytorch_lightning as L


class BasicModel(Module):
    def __init__(self):
        super().__init__()

        self.conv = Sequential(
            LazyConv2d(32, 3, padding=0),  # strictly zero padding on input
            ReLU(),
            LazyBatchNorm2d(),
            LazyConv2d(64, 3),
            ReLU(),
            LazyBatchNorm2d(),
            LazyConv2d(128, 3),
            ReLU(),
            LazyBatchNorm2d(),
            LazyConv2d(128, 3),
            ReLU(),
            LazyBatchNorm2d(),
            LazyConv2d(128, 3),
            ReLU(),
            LazyConv2d(128, 3),
            ReLU(),
            LazyBatchNorm2d(),
        )

        self.fc = Sequential(
            LazyLinear(256),
            ReLU(),
            LazyBatchNorm1d(),
            LazyLinear(128),
            ReLU(),
            LazyBatchNorm1d(),
            LazyLinear(len(ACTIONS)),
        )

    def forward(self, map, bomb):
        x = self.conv(map)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, bomb.reshape(-1, 1)), dim=1)
        x = self.fc(x)

        return softmax(x, dim=1)


class SkipCoordsModel(Module):
    def __init__(self):
        super().__init__()

        self.conv_list = ModuleList(
            [
                # each block should keep H, W constant, so that input can be skipcon'd
                Sequential(
                    # no padding, otherwise padding can be confused with an empty square:
                    # I wish Conv2d supported constant non-zero padding...
                    LazyConv2d(32, 3, padding=0),
                    ReLU(),
                    LazyBatchNorm2d(),
                    LazyConv2d(64, 3, padding=2),
                    ReLU(),
                    LazyBatchNorm2d(),
                ),
                Sequential(
                    LazyConv2d(128, 3, padding=1),
                    ReLU(),
                    LazyBatchNorm2d(),
                ),
                Sequential(
                    LazyConv2d(128, 3, padding=1),
                    ReLU(),
                    LazyBatchNorm2d(),
                ),
                Sequential(
                    LazyConv2d(128, 3, padding=1),
                    ReLU(),
                    LazyBatchNorm2d(),
                ),
                Sequential(
                    LazyConv2d(128, 3, padding=1),
                    ReLU(),
                    LazyBatchNorm2d(),
                ),
            ]
        )

        self.fc = Sequential(
            LazyLinear(256),
            ReLU(),
            LazyBatchNorm1d(),
            LazyLinear(128),
            ReLU(),
            LazyBatchNorm1d(),
            LazyLinear(len(ACTIONS)),
        )

    def forward(self, map, bomb):
        # coords = map[:, -2:, :, :]
        x = skipper(map, map, self.conv_list)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, bomb.reshape(-1, 1)), dim=1)
        x = self.fc(x)

        return softmax(x, dim=1)


def skipper(start: Tensor, skip: Tensor, layers: ModuleList):
    res = layers[0](start)
    for layer in layers[1:]:  # pyright: ignore
        res = torch.cat((res, skip), dim=1)
        res = layer(res)
    return res


class Lighter(L.LightningModule):
    def __init__(self, model, total_steps):
        super().__init__()
        self.model = model
        self.total_steps = total_steps

    def training_step(self, batch, batch_idx):
        map, bomb, action = batch
        out = self.model(map, bomb)
        action = action
        loss = cross_entropy(out, action)

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
