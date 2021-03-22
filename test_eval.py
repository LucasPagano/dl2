import os
import shutil
import sys
from pathlib import Path
from torchvision.utils import save_image
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import wandb
from utils import Dotdict
import pytorch_fid
from zoo import BVAE, InfoGAN

import numpy as np
import copy

run_id = "11x2nzv1"
nb_examples = 5

wandb.init(project="eval-infogan")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
api = wandb.Api()
run = api.run("lucas_p/infogan/{}".format(run_id))
config = Dotdict(run.config)
print(config)

state = torch.load("./models/{}/model.pt".format(run_id))
print(state.keys())
print("Best epoch : {}".format(state["epoch"]))
model = InfoGAN(config).to(device).eval()
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


def plot_mus():
    print("starting to plot mus")
    columns = ['mu1', 'mu2', 'class']
    mus = pd.DataFrame(columns=columns)
    for batch_images, classes_real in dataloader:
        batch_images, classes_real = batch_images.to(device), classes_real.to(device)
        classes = model.classifier(batch_images)
        encoded = torch.cat((model.encode(batch_images), classes), dim=1)
        encoded = model.mu_logvar(encoded)
        mu = encoded[:, :model.z_dim]
        to_add = torch.hstack((mu, classes_real.unsqueeze(-1)))
        mus = mus.append(pd.DataFrame(to_add.detach().cpu().numpy(), columns=columns), ignore_index=True)

    mus = mus.astype({"class": "category"})
    fig, axes = plt.subplots()
    sns.scatterplot(ax=axes, data=mus, x="mu1", y="mu2", hue="class")
    wandb.log({"mu1 vs mu2": wandb.Image(plt)})


def plot_test_set():
    cpt = 0
    imgs = []
    for batch_images, classes_real in dataloader:
        batch_images, classes_real = batch_images.to(device), classes_real.to(device)
        classes_pred = model.classifier(batch_images)
        classes_pred = classes_pred[0].exp().repeat(10, 1)
        print(classes_pred)
        classes_pred = np.zeros(10)
        classes_pred[np.random.randint(0, 10)] = 1
        classes_pred = torch.Tensor(classes_pred).to(device).repeat(10, 1)

        test_latents_c1 = np.zeros(shape=(10, config.latent_size))
        test_latents_c1[:, 0] = np.linspace(-100, 10, 10)
        test_latents_c1 = torch.FloatTensor(test_latents_c1).to(device)
        test_latents_c1 = torch.hstack((test_latents_c1, classes_pred))
        x_save1 = model.decode(test_latents_c1)
        imgs.append((torchvision.utils.make_grid(x_save1, nrow=10)))
        cpt += 1
        if cpt == nb_examples:
            wandb.log({"im": wandb.Image(torchvision.utils.make_grid(imgs, nrow=1))})
            sys.exit(0)


def generate_vae(n=10000):
    out_folder = "out/" + str(run_id)
    shutil.rmtree(out_folder, ignore_errors=True)
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    out_mnist = "datasets/MNIST/full{}".format(n // 1000)
    cpt = 0
    if not os.path.exists(out_mnist):
        print("Saving mnist test images")
        Path(out_mnist).mkdir(parents=True, exist_ok=True)
        for i, (batch_images, _) in enumerate(dataloader):
            for j in range(batch_images.size(0)):
                save_image(batch_images[j], os.path.join(out_mnist, "{}.png".format(cpt)))

    print("Starting generation")
    all_images = []
    batch_size = config.batch_size
    while len(all_images) < n:
        # generate batch size examples
        # pick from random distribution
        z = torch.randn(batch_size, config.latent_size).to(device)
        # add class
        one_hot = torch.zeros(batch_size, 10).to(device)
        index = torch.randint(10, (batch_size, 1)).to(device)
        one_hot = one_hot.scatter(1, index, 1)
        z = torch.cat((z, one_hot), dim=1)
        # generate images
        gen = model.decode(z)
        gen = gen.detach().cpu()
        all_images.extend([gen[x] for x in range(gen.size(0))])

    print("Generation done, saving images..")
    for i, image in enumerate(all_images):
        file_name = os.path.join(out_folder, "{}.png".format(i))
        save_image(image, file_name)
        if i < 10:
            wandb.log({"img/{}".format(i): wandb.Image(image)})
        if i == n:
            break


def generate_infoGAN(n=10000):
    out_folder = "out/" + str(run_id)
    shutil.rmtree(out_folder, ignore_errors=True)
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    print("Starting generation")
    all_images = []
    batch_size = config.batch_size
    while len(all_images) < n:
        # generate batch size examples
        # pick from random distribution
        fix_noise = torch.Tensor(config.batch_size, 62).uniform_(-1, 1).to(device)
        z = torch.randn(batch_size, 2).to(device)
        # add class
        one_hot = torch.zeros(batch_size, 10).to(device)
        index = torch.randint(10, (batch_size, 1)).to(device)
        one_hot = one_hot.scatter(1, index, 1)
        z = torch.cat([fix_noise, one_hot, z], dim=1).view(batch_size, 74, 1, 1)
        # generate images
        gen = model.generator(z)
        gen = gen.detach().cpu()
        all_images.extend([gen[x] for x in range(gen.size(0))])

    print("Generation done, saving images..")
    for i, image in enumerate(all_images):
        file_name = os.path.join(out_folder, "{}.png".format(i))
        save_image(image, file_name)
        if i < 10:
            wandb.log({"img/{}".format(i): wandb.Image(image)})
        if i == n:
            break


generate_infoGAN()
