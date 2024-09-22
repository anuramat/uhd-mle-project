#!/usr/bin/env python


import argparse
from agent_code.vkl.data import TranDataset
from os import listdir, cpu_count
from os.path import exists, join
from re import search
from torch import load, save
from torch.utils.data import DataLoader
import agent_code.vkl.models as models
import pytorch_lightning as L
import torch

# from pytorch_lightning.tuner.tuning import Tuner


parser = argparse.ArgumentParser()
parser.add_argument("--n-epochs", type=int, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--n-workers", type=int, default=cpu_count())
parser.add_argument("--batch-size", type=int, default=5120)
args = parser.parse_args()

# parameters
torch.set_float32_matmul_precision("medium")
precision = "16-mixed"
trans_dir = "agent_code/watcher/data/"

# data
trans_filenames = [f for f in listdir(trans_dir) if search(r".*\.pt", f)]
print(f"{len(trans_filenames)} data files found: \n\t {', '.join(trans_filenames)}")
paths = [join(trans_dir, f) for f in trans_filenames]
print("Loading files...")
trans = sum([load(path, weights_only=False) for path in paths], [])
print(f"{len(trans)} transitions loaded.")
dataset = TranDataset(trans)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.n_workers,
    pin_memory=True,
    # persistent_workers=True, # WARN causes a memory leak
    drop_last=True,
)
del trans

# raw model
raw_model = None
if args.input not in ("none"):
    if not exists(args.input):
        raise FileNotFoundError("Input model file doesn't exist")
    print("Loading existing model")
    raw_model = load(args.input, weights_only=False)
else:
    print("Starting training from scratch")
    raw_model = models.MyBelovedCNN()
    # HACK workaround for the tuner bug with the lazy layers: initialize manually
    map, aux, _, _ = next(iter(dataloader))

total_steps = args.n_epochs * len(dataloader)

# lightning
raw_model.train()
model = models.Lighter(raw_model, total_steps=total_steps)
trainer = L.Trainer(accelerator="gpu", max_epochs=args.n_epochs, precision=precision)
# # TODO figure out how to make tuner ignore the scheduler
# tuner = Tuner(trainer)
# tuner.lr_find(model, dataloader)
trainer.fit(model=model, train_dataloaders=dataloader)

# save the result
save(raw_model, args.output)
