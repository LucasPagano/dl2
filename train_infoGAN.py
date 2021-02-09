import torch
import numpy as np

import wandb
import zoo

HPP_DEFAULT = dict(
    batch_size=512,
    val_batch_size=256,
    epochs=150,
    lr=5e-4,
    no_cuda=False,
    seed=42,
    beta=4,
    latent_size=10,
    optimizer="Adam"
)

run = wandb.init(project="infogan", config=HPP_DEFAULT)
config = wandb.config
print(config)
torch.manual_seed(config.seed)
np.random.seed(config.seed)

use_cuda = not config.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

model = zoo.InfoGAN(config).to(device)
wandb.watch(model, log="all")
for name, param in model.named_parameters():
    print(name, param.data.mean(), param.data.std())