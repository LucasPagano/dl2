import torch
import numpy as np
import torchvision
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb
import zoo

HPP_DEFAULT = dict(
    batch_size=128,
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

use_cuda = not config.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

model = zoo.InfoGAN(config).to(device)
wandb.watch(model, log="all")

optimD = optim.Adam([{'params': model.q_disc_front_end.parameters()},
                     {'params': model.discriminator_head.parameters()}], lr=0.0002, betas=(0.5, 0.99))
optimG = optim.Adam([{'params': model.generator.parameters()}, {'params': model.q_head.parameters()}], lr=0.001,
                    betas=(0.5, 0.99))

dataset = torchvision.datasets.MNIST('./datasets', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)

criterionD = nn.BCELoss()
criterionQ_dis = nn.CrossEntropyLoss()
criterionQ_con = model.log_gaussian

real_x = Variable(torch.FloatTensor(config.batch_size, 1, 28, 28).to(device))
label = Variable(torch.FloatTensor(config.batch_size, 1).to(device), requires_grad=False)
dis_c = Variable(torch.FloatTensor(config.batch_size, 10).to(device))
con_c = Variable(torch.FloatTensor(config.batch_size, 2).to(device))
noise = Variable(torch.FloatTensor(config.batch_size, 62).to(device))

# fixed random variables
c = np.linspace(-1, 1, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)
c1 = np.hstack([c, np.zeros_like(c)])
c2 = np.hstack([np.zeros_like(c), c])
idx = np.arange(10).repeat(10)
one_hot = np.zeros((100, 10))
one_hot[range(100), idx] = 1
fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)



