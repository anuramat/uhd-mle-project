# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import agent_code.vkl.models as models
from agent_code.vkl.data import TranDataset
from torch.utils.data import DataLoader
from torch import load, save
import pytorch_lightning as L
import torch
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from gc import collect

# %% [markdown]
# # Parameters

# %%
epochs = 1024
batch_size = 512
torch.set_float32_matmul_precision("medium")
precision = "16-mixed"

# %% [markdown]
# # Data

# %%
# join 4 datasets
dir = "agent_code/watcher/output/"
filenames = [f for f in listdir(dir) if f[-3:] == ".pt"][:1]
paths = [join(dir, f) for f in filenames]
trans = sum([load(path, weights_only=False) for path in paths], [])

dataset = TranDataset(trans)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=32,
    pin_memory=True,
    persistent_workers=True,
)
del trans
collect()

# %%
len(dataset)

# %%
plt.imshow(dataset[0][0][0, ...])

# %%
plt.imshow(dataset[0][0][-1, ...])

# %% [markdown]
# # Model definition

# %%
raw_model = models.MyBelovedCNN()
total_steps = epochs * len(dataloader)
model = models.Lighter(raw_model, total_steps=total_steps)
# model = torch.compile(model) # doesn't work for now

# %% [markdown]
# # Training

# %% editable=true slideshow={"slide_type": ""}
trainer = L.Trainer(accelerator="gpu", max_epochs=epochs, precision=precision)
try:
    trainer.fit(model=model, train_dataloaders=dataloader)
except KeyboardInterrupt:
    print("stopped early on C-C")

# %% [markdown]
# # Save the model

# %%
save(raw_model, "agent_code/vkl/output/model.pt")
