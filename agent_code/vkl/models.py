import torch
import torch.nn as nn
import torch.nn.functional as F
from agent_code.vkl.consts import N_CHANNELS, ACTIONS
import pytorch_lightning as L


class BasicModel(nn.Module):
    def __init__(self, radius):
        super().__init__()

        # (!) zero or 1 padding
        conv_layers = (
            # TODO shift map values by min+1, add padding=1
            # intuition: 0 means idk
            # for now we could just compensate with a bigger fov radius
            nn.Conv2d(N_CHANNELS, 10, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(40, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(*conv_layers)

        # calculate total h/w decrease
        offset = sum(
            [
                2 if type(i) is nn.Conv2d and i.padding[0] == i.padding[1] == 0 else 0
                for i in conv_layers
            ]
        )
        orig_dim = radius * 2 + 1
        out_channels = conv_layers[-2].out_channels
        final_dim = orig_dim - offset
        print(f"final conv layer size: {final_dim}")
        self.conv_output_dim = out_channels * final_dim**2

        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
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
        action = action
        loss = F.cross_entropy(out, action)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Adam is for some reason invisible to pyright
        # <https://github.com/pytorch/pytorch/issues/134985>
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)  # pyright: ignore
        return optimizer
