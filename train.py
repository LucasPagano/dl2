import copy
import datetime
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision
import wandb

import zoo

HPP_DEFAULT = dict(
    batch_size=512,
    val_batch_size=256,
    epochs=50,
    lr=5e-4,
    no_cuda=False,
    seed=42,
    beta=4,
    latent_size=10
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

use_cuda = not config.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

### WANDB
# plot model graph
model = zoo.BVAE(beta=config.beta, z_dim=config.latent_size).to(device)
wandb.watch(model, log="all")

model_dir = os.path.join("./models", folder_name)
shutil.rmtree(model_dir, ignore_errors=True)
Path(model_dir).mkdir(parents=True, exist_ok=True)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()])

dataset = torchvision.datasets.MNIST(
    root="./datasets/", train=True, transform=transform, download=True
)
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=config.batch_size, shuffle=True, **kwargs
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=config.val_batch_size, shuffle=True, **kwargs
)
best_val_loss = math.inf

log_images = []
# training
for epoch in range(1, config.epochs + 1):
    total_train_loss = 0
    model.train()
    for batch_images, _ in train_loader:
        optimizer.zero_grad()
        batch_images = batch_images.to(device)
        reconstructed, mu, logvar = model(batch_images)
        train_loss = model.get_loss(reconstructed, batch_images, mu, logvar)
        total_train_loss += train_loss
        train_loss.backward()
        optimizer.step()

    ## WANDB
    to_log = {"Loss/train": total_train_loss / len(train_loader)}

    # compute validation loss and save model
    with torch.no_grad():
        model.eval()
        total_val_loss = 0
        total_val_mse = 0
        mse = torch.nn.MSELoss()
        for batch_images, _ in val_loader:
            batch_images = batch_images.to(device)
            reconstructed, mu, logvar = model(batch_images)
            val_loss = model.get_loss(reconstructed, batch_images, mu, logvar)
            total_val_loss += val_loss
            total_val_mse += mse(reconstructed.view(-1, 32 * 32), batch_images.view(-1, 32 * 32))
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss

            # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
            torch.save(model.state_dict(), os.path.join("./models", folder_name + ".pt"))
            wandb.save(os.path.join("./models", folder_name + ".pt"))

        ### WANDB
        to_log["Loss/val"] = total_val_loss / len(val_loader)
        to_log["Mse/val"] = total_val_mse / len(val_loader)

        # plot images
        if epoch % 10 == 0:
            print("Epoch {}/{}".format(epoch, config.epochs))
            batch_images, reconstructed = batch_images[:4, :, :], reconstructed[:4, :, :]
            img_grid_inputs_test = torchvision.utils.make_grid(batch_images)
            img_grid_outputs_test = torchvision.utils.make_grid(reconstructed.view(-1, 32, 32).unsqueeze(1))

            ### WANDB
            images_to_log = [wandb.Image(img_grid_inputs_test), wandb.Image(img_grid_outputs_test)]
            to_log["images/epoch{}".format(epoch)] = copy.deepcopy(images_to_log)
            log_images.extend(images_to_log)

    wandb.log(to_log)

print(log_images)
wandb.log({"img_all": log_images})