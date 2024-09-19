import torch
from torch import Tensor, arange
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
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.adamw import AdamW
import agent_code.vkl.typing as T
import pytorch_lightning as L


class MyBelovedCNN(Module):
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
            LazyLinear(len(T.ACTIONS)),
        )

    def forward(self, map, aux):
        x = skipper(map, map, self.conv_list)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, aux), dim=1)
        x = self.fc(x)

        return x


def skipper(start: Tensor, skip: Tensor, layers: ModuleList):
    res = layers[0](start)
    for layer in layers[1:]:  # pyright: ignore
        res = torch.cat((res, skip), dim=1)
        res = layer(res)
    return res


class Lighter(L.LightningModule):
    def __init__(self, model, total_steps, lr=3e-4):
        super().__init__()
        self.model = model
        self.total_steps = total_steps
        self.lr = lr

    def training_step(self, batch, batch_idx):
        map, aux, action, reward = batch
        preds_raw = self.model(map, aux)

        n = map.shape[0]
        preds = preds_raw[arange(n), action]
        loss = mse_loss(preds, reward)

        self.log("train_loss", int(loss))
        self.log("lr", self.lr_schedulers().get_last_lr()[0])  # pyright:ignore
        return loss

    def configure_optimizers(self):  # pyright:ignore
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = OneCycleLR(
            optimizer=optimizer, max_lr=self.lr, total_steps=self.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
