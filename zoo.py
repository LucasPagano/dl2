import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Conv2DReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilatation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilatation)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class BVAE(nn.Module):
    def __init__(self, beta=None, z_dim=10, nc=1):
        super(BVAE, self).__init__()
        self.name = "BVAE"
        if beta is None:
            self.beta = 1
        else:
            self.beta = beta
        self.z_dim = z_dim

        self.encoder = nn.Sequential(

            Conv2DReLU(nc, 32, kernel_size=4, stride=2, padding=1),  # (B, nc, 32, 32) -> (B, 32, 16, 16)
            Conv2DReLU(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 32, 32, 32) -> (B, 64, 8, 8)
            Conv2DReLU(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 64, 8, 8) -> (B, 128, 4, 4)
            Conv2DReLU(128, 256, kernel_size=4, stride=1),  # (B, 128, 4, 4) -> (B, 256, 1, 1)
            View((-1, 256)),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim * 2)
        )
        self.weight_init()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            View((-1, 256, 1, 1)),
            nn.ConvTranspose2d(256, 64, 4),  # (B, 256, 1, 1) -> (B, 64, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # (B, 64, 4, 4) -> (B, 64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 64, 8, 8) -> (B, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # (B, 32, 16, 16) -> (B, nc, 32, 32)
            nn.Sigmoid()
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def encode(self, x):
        return self.encoder(x)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        mu, log_var = encoded[:, :self.z_dim], encoded[:, self.z_dim:]
        z = self.sampling(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def get_loss(self, recon_x, x, mu, log_var):
        bce = F.binary_cross_entropy(recon_x.view(-1, 32 * 32), x.view(-1, 32 * 32), reduction='sum')
        # mse = torch.nn.MSELoss()(recon_x.view(x.size()), x)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # divide by batch size
        return bce + kld * self.beta


class InfoGAN(nn.Module):

    def __init__(self, config):
        super(InfoGAN, self).__init__()
        self.name = "InfoGAN"
        self.generator = self.Generator(config)
        self.q_disc_front_end = self.FrontEnd(config)
        self.discriminator_head = nn.Sequential(
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        ).apply(self.weights_init)
        self.q_head = self.QHead(config)
        self.apply(self.weights_init)

    class Generator(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.main = self.generator = nn.Sequential(
                nn.ConvTranspose2d(74, 1024, 1, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
                nn.Sigmoid())

        def forward(self, x):
            return self.main(x)

    # Front end for discriminator and Q
    class FrontEnd(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(128, 1024, 7, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
            )

        def forward(self, x):
            return self.main(x)

    class QHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.conv = nn.Conv2d(1024, 128, 1, bias=False)
            self.bn = nn.BatchNorm2d(128)
            self.lReLU = nn.LeakyReLU(0.1, inplace=True)
            self.conv_disc = nn.Conv2d(128, 10, 1)
            self.conv_mu = nn.Conv2d(128, 2, 1)
            self.conv_var = nn.Conv2d(128, 2, 1)

        def forward(self, x):
            y = self.conv(x)

            disc_logits = self.conv_disc(y).squeeze()

            mu = self.conv_mu(y).squeeze()
            var = self.conv_var(y).squeeze().exp()

            return disc_logits, mu, var

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
