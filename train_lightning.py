import os
import shutil
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

import wandb
import zoo
from datasets import MNISTDataModule

HPP_DEFAULT = dict(
    batch_size=512,
    val_batch_size=256,
    epochs=150,
    lr=5e-4,
    no_cuda=False,
    seed=42,
    beta=4,
    latent_size=10,
    nc=1
)

### WANDB
# init run and get config for sweep initialized runs
run = wandb.init(project="wandb-demo", config=HPP_DEFAULT)
config = wandb.config
print(config)
torch.manual_seed(config.seed)
np.random.seed(config.seed)

### WANDB
# rename run folder
folder_name = "BVAE-b{}_".format(config.beta) + "z{}".format(config.latent_size)
run.name = folder_name

### WANDB
# plot model graph
model = zoo.BVAE(config)
wandb.watch(model, log="all")

model_dir = os.path.join("./models", folder_name)
shutil.rmtree(model_dir, ignore_errors=True)
Path(model_dir).mkdir(parents=True, exist_ok=True)

## Pytorch Lightning
wandb_logger = pl.loggers.WandbLogger()
trainer = pl.Trainer(gpus=-1, logger=wandb_logger)
data_module = MNISTDataModule(config)
trainer.fit(model, data_module)

