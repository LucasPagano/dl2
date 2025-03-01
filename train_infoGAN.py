import os
import shutil
from pathlib import Path

import torch
import numpy as np
import torchvision
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
import wandb
import zoo

HPP_DEFAULT = dict(
    lambda_loss=0.1,
    batch_size=100,
    val_batch_size=256,
    epochs=100,
    no_cuda=False,
    seed=42,
)

run = wandb.init(project="infogan", config=HPP_DEFAULT)
config = wandb.config
print(config)
torch.manual_seed(config.seed)
np.random.seed(config.seed)

model_dir = os.path.join("./models", run.id)
shutil.rmtree(model_dir, ignore_errors=True)
Path(model_dir).mkdir(parents=True, exist_ok=True)

use_cuda = not config.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

infoGAN = zoo.InfoGAN(config).to(device)
wandb.watch(infoGAN, log="all")

optimD = optim.Adam([{'params': infoGAN.q_disc_front_end.parameters()},
                     {'params': infoGAN.discriminator_head.parameters()}], lr=0.0002, betas=(0.5, 0.99))
optimG = optim.Adam([{'params': infoGAN.generator.parameters()}, {'params': infoGAN.q_head.parameters()}], lr=0.001,
                    betas=(0.5, 0.99))

dataset = torchvision.datasets.MNIST('./datasets', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=True, shuffle=True, **kwargs)

criterionD = nn.BCELoss()
criterionQ_dis = nn.CrossEntropyLoss()
criterionQ_con = infoGAN.log_gaussian

real_x = Variable(torch.FloatTensor(config.batch_size, 1, 28, 28).to(device))
label = Variable(torch.FloatTensor(config.batch_size, 1).to(device), requires_grad=False)
dis_c = Variable(torch.FloatTensor(config.batch_size, 10).to(device))
con_c = Variable(torch.FloatTensor(config.batch_size, 2).to(device))
noise = Variable(torch.FloatTensor(config.batch_size, 62).to(device))

# fixed random variables
## due to the size of idx, batch size has to be 100
c = np.linspace(-1, 1, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)
c1 = np.hstack([c, np.zeros_like(c)])
c2 = np.hstack([np.zeros_like(c), c])
idx = np.arange(10).repeat(10)
one_hot = np.zeros((config.batch_size, 10))
one_hot[range(config.batch_size), idx] = 1
fix_noise = torch.Tensor(config.batch_size, 62).uniform_(-1, 1)

images_log_all = []
for epoch in range(config.epochs):
    print("Epoch {}/{}".format(epoch, config.epochs))
    total_G_loss = 0
    total_D_loss = 0
    for num_iters, batch_data in enumerate(dataloader, 0):
        ## Discriminator part : maximize log(D(x)) + log(1 - D(G(z)))
        # real part
        optimD.zero_grad()

        x, _ = batch_data

        real_x.data.copy_(x)
        fe_out1 = infoGAN.q_disc_front_end(real_x)
        probs_real = infoGAN.discriminate(fe_out1)
        label.data.fill_(1)
        loss_real = criterionD(probs_real, label)
        loss_real.backward()

        # fake part
        z, idx = utils.noise_sample(dis_c, con_c, noise, config.batch_size)
        fake_x = infoGAN.generator(z)
        fe_out2 = infoGAN.q_disc_front_end(fake_x.detach())
        probs_fake = infoGAN.discriminate(fe_out2)
        label.data.fill_(0)
        loss_fake = criterionD(probs_fake, label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake
        total_D_loss += D_loss

        optimD.step()

        ## G and Q part
        # only fake : trivial for G, Q : can only do during fake bc codes aren't known during real
        # fake data is treated as real (label is 1)
        optimG.zero_grad()

        fe_out = infoGAN.q_disc_front_end(fake_x)
        probs_fake = infoGAN.discriminate(fe_out)
        label.data.fill_(1)

        reconstruct_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = infoGAN.q_head(fe_out)
        class_ = torch.LongTensor(idx).to(device)
        target = Variable(class_)
        dis_loss = criterionQ_dis(q_logits, target)
        con_loss = criterionQ_con(con_c, q_mu, q_var) * config.lambda_loss

        G_loss = reconstruct_loss + dis_loss + con_loss
        total_G_loss += G_loss

        G_loss.backward()
        optimG.step()

    ## logs generated image
    ## WANDB
    to_log = {
        "Loss/G": total_G_loss / len(dataloader),
        "Loss/D": total_D_loss / len(dataloader),
    }
    # log images every 10 epochs
    if epoch % 10 == 0:
        noise.data.copy_(fix_noise)
        dis_c.data.copy_(torch.Tensor(one_hot))
        con_c.data.copy_(torch.from_numpy(c1))
        z1 = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
        con_c.data.copy_(torch.from_numpy(c2))
        z2 = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
        with torch.no_grad():
            x_save1 = infoGAN.generator(z1)
            x_save2 = infoGAN.generator(z2)
        grid1 = wandb.Image(torchvision.utils.make_grid(x_save1, nrow=10))
        grid2 = wandb.Image(torchvision.utils.make_grid(x_save2, nrow=10))

        to_log["images/epoch{}_c1".format(epoch)] = grid1
        to_log["images/epoch{}_c2".format(epoch)] = grid2
        images_log_all.extend([grid1, grid2])

        # save model
        torch.save({
            "epoch": epoch,
            "state_dict": infoGAN.state_dict(),
        }, os.path.join(model_dir, "model{}.pt".format(epoch)))
    wandb.log(to_log)
wandb.log({"img_all": images_log_all})
