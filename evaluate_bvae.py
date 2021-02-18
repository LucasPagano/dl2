import torch
import torchvision
import matplotlib.pyplot as plt
import wandb

from zoo import BVAE
import numpy as np
import copy

wandb.init(project="eval-vae", entity="lucas_p")

z_dim = 16

model = BVAE(z_dim=z_dim, beta=27).eval()
model.load_state_dict(torch.load("./models/BVAE-b27_z16.pt"))

# test two first channels
test_latents_c1 = np.random.uniform(low=-1, high=1, size=(z_dim, 10))
test_latents_c2 = copy.copy(test_latents_c1)
test_latents_c1[0] = np.linspace(-1, 1, 10)
test_latents_c2[1] = np.linspace(-1, 1, 10)
test_latents_c1 = torch.FloatTensor(test_latents_c1).permute(1, 0)
test_latents_c2 = torch.FloatTensor(test_latents_c2).permute(1, 0)
with torch.no_grad():
    x_save1 = model.decode(test_latents_c1)
    x_save2 = model.decode(test_latents_c2)

grid1 = wandb.Image(torchvision.utils.make_grid(x_save1, nrow=10))
grid2 = wandb.Image(torchvision.utils.make_grid(x_save2, nrow=10))
logdict = {"i1": grid1, "i2": grid2}
wandb.log(logdict)
