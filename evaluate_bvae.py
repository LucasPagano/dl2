import torch
import torchvision
import matplotlib.pyplot as plt
import wandb
from utils import Dotdict

from zoo import BVAE
import numpy as np
import copy

run_id = "p0zusi04"
nb_examples = 5

wandb.init(project="eval-vae", entity="lucas_p")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
api = wandb.Api()
run = api.run("lucas_p/wandb-demo/{}".format(run_id))
config = Dotdict(run.config)
print(config)

state = torch.load("./models/{}/model.pt".format(run_id))
config.latent_size = state["index_latent"]
model = BVAE(config).to(device).eval()
model.load_state_dict(state["state_dict"])

# test two first channels
test_latents_c1 = np.random.normal(size=(config.latent_size, 10))
test_latents_c2 = copy.copy(test_latents_c1)
test_latents_c1[0] = np.linspace(-1, 1, 10)
test_latents_c2[1] = np.linspace(-1, 1, 10)
test_latents_c1 = np.repeat(test_latents_c1, nb_examples, axis=0)
test_latents_c2 = np.repeat(test_latents_c1, nb_examples, axis=0)
test_latents_c1 = torch.FloatTensor(test_latents_c1).permute(1, 0).to(device)
test_latents_c2 = torch.FloatTensor(test_latents_c2).permute(1, 0).to(device)
with torch.no_grad():
    x_save1 = model.decode(test_latents_c1)
    x_save2 = model.decode(test_latents_c2)

grid1 = wandb.Image(torchvision.utils.make_grid(x_save1, nrow=10))
grid2 = wandb.Image(torchvision.utils.make_grid(x_save2, nrow=10))

logdict = {"i1": grid1, "i2": grid2}
wandb.log(logdict)
