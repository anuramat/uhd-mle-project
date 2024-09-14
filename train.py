#!/usr/bin/env python

from agent_code.vkl.data import TranDataset
from gc import collect
from os import listdir
from os.path import exists, join
from re import search
from torch import load, save
from torch.utils.data import DataLoader
import agent_code.vkl.models as models
import pytorch_lightning as L
import sys
import torch

# from pytorch_lightning.tuner.tuning import Tuner

load_existing = True
if len(sys.argv) > 1 and sys.argv[1] == "new":
    load_existing = False

# parameters
epochs = 2000
batch_size = 1024
torch.set_float32_matmul_precision("medium")
precision = "16-mixed"
num_workers = 32
source_model = "source_model.pt"
result_model_path = "result_model.pt"
trans_dir = "agent_code/watcher/output/"

# ./datagen.sh 1024
trans_filenames = [f for f in listdir(trans_dir) if search(r"trans.*\.pt", f)]
paths = [join(trans_dir, f) for f in trans_filenames]
trans = sum([load(path, weights_only=False) for path in paths], [])
dataset = TranDataset(trans)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
del trans
collect()

# raw model
raw_model = None
if load_existing:
    if not exists(source_model):
        raise FileNotFoundError("Source model file doesn't exist")
    print("Loading existing model")
    raw_model = load(source_model)
else:
    print("Starting training from scratch")
    raw_model = models.MyBelovedCNN()
    # HACK workaround for the tuner bug with the lazy layers: initialize manually
    map, aux, _, _ = next(iter(dataloader))

total_steps = epochs * len(dataloader)

# lightning
model = models.Lighter(raw_model, total_steps=total_steps)
trainer = L.Trainer(accelerator="gpu", max_epochs=epochs, precision=precision)
# # TODO figure out how to make tuner ignore the scheduler
# tuner = Tuner(trainer)
# tuner.lr_find(model, dataloader)
trainer.fit(model=model, train_dataloaders=dataloader)

# save the result
save(raw_model, result_model_path)
