import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from agent_code.vkl.consts import N_CHANNELS, ACTIONS
import pytorch_lightning as L


class BasicModel(nn.Module):
    def __init__(self, radius):
        super().__init__()

        # (!) zero or 1 padding
        conv_list = [
            # TODO shift map values by min+1, add padding=1
            # intuition: 0 means idk
            # for now we could just compensate with a bigger fov radius
            nn.Conv2d(N_CHANNELS, 32, kernel_size=3, stride=1, padding=0),
            # this one is kinda special yeah
            nn.ReLU(),
            64,
            nn.ReLU(),
            nn.AvgPool2d(2),
            128,
            nn.ReLU(),
            128,
            nn.ReLU(),
            nn.AvgPool2d(2),
            128,
            nn.ReLU(),
            128,
            nn.ReLU(),
            nn.AvgPool2d(2),
        ]

        out_channels = 0
        for i, v in enumerate(conv_list):
            if type(v) is int:
                conv_list[i] = nn.Conv2d(
                    out_channels, v, kernel_size=3, stride=1, padding=1
                )
                out_channels = v
            if type(v) is nn.Conv2d:
                out_channels = v.out_channels

        self.conv = nn.Sequential(*conv_list)

        # calculate output dim
        orig_side = radius * 2 + 1
        fake_input = torch.zeros(1, 5, orig_side, orig_side)
        fake_output = self.conv(fake_input)
        side = fake_output.shape[-1]
        print(f"final conv layer side: {side}")
        self.conv_output_dim = out_channels * side**2

        fc_list = [
            256,
            nn.ReLU(),
            128,
            nn.ReLU(),
        ]

        out_features = self.conv_output_dim + 1
        for i, v in enumerate(fc_list):
            if type(v) is int:
                fc_list[i] = nn.Linear(out_features, v)
                out_features = v
            if type(v) is nn.Linear:
                out_features = v.out_features
        fc_list.append(nn.Linear(out_features, len(ACTIONS)))
        self.fc = nn.Sequential(*fc_list)

    def forward(self, map, bombful):
        x = self.conv(map)

        x = x.reshape(-1, self.conv_output_dim)
        x = torch.cat((x, bombful.reshape(-1, 1)), dim=1)

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

        # opt = self.optimizers()
        sch = self.lr_schedulers()

        self.log("train_loss", loss)
        self.log("lr", sch.get_last_lr()[0])  # pyright:ignore
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
