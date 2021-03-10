import sys

import torch
import torchvision
import matplotlib.pyplot as plt
import wandb
from utils import Dotdict

from zoo import BVAE
import numpy as np
import copy

run_id = "2pjycujg"
nb_examples = 5

wandb.init(project="eval-vae", entity="lucas_p")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
api = wandb.Api()
run = api.run("lucas_p/wandb-demo/{}".format(run_id))
config = Dotdict(run.config)
print(config)

state = torch.load("./models/{}/model.pt".format(run_id))
print(state.keys())
print("Best epoch : {}".format(state["epoch"]))
model = BVAE(config).to(device).eval()
model.load_state_dict(state["state_dict"])
wandb.watch(model)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()])

dataset = torchvision.datasets.MNIST(
    root="./datasets/", train=False, transform=transform, download=True
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.batch_size, shuffle=True
)

cpt = 0
imgs = []
for batch_images, classes_real in dataloader:
    batch_images, classes_real = batch_images.to(device), classes_real.to(device)
    classes_pred = model.classifier(batch_images)
    classes_pred = classes_pred[0].repeat(10, 1)
    test_latents_c1 = np.zeros(shape=(10, config.latent_size))
    test_latents_c1[:, 0] = np.linspace(-10, 10, 10)
    test_latents_c1 = torch.FloatTensor(test_latents_c1).to(device)
    test_latents_c1 = torch.hstack((test_latents_c1, classes_pred))
    x_save1 = model.decode(test_latents_c1)
    imgs.append((torchvision.utils.make_grid(x_save1, nrow=10)))
    cpt += 1
    if cpt == nb_examples:
        wandb.log({"im":wandb.Image(torchvision.utils.make_grid(imgs))})
        sys.exit(0)
