import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import numpy as np


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
    def __init__(self, config):
        super(BVAE, self).__init__()
        self.name = "BVAE"
        self.image_size = config.image_size if "image_size" in config.keys() else 32
        self.conditional = config.conditional if "conditional" in config.keys() else False
        self.beta = config.beta if "beta" in config.keys() else 1
        self.in_channels = config.nc if "nc" in config.keys() else 1
        self.hidden_dims = config.hidden_dims if "hidden_dims" in config.keys() else [32, 64, 128, 256, 512]
        self.z_dim = config.latent_size
        self.classes_dim = 10

        # build encoder
        modules = []
        in_channels = self.in_channels

        if self.conditional:
            self.embed_class = nn.Linear(self.classes_dim, self.image_size * self.image_size)
            self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            in_channels += 1

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.z_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.z_dim)

        # build decoder
        modules = []
        if self.conditional:
            self.decoder_input = nn.Linear(self.z_dim + self.classes_dim, self.hidden_dims[-1])
        else:
            self.decoder_input = nn.Linear(self.z_dim, self.hidden_dims[-1])

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1],
                               self.hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

        # init weights
        # self.weight_init()

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
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x, classes_real):
        if self.conditional:
            # make one hot from labels
            one_hot = torch.zeros(classes_real.size(0), self.classes_dim).to(classes_real.device)
            one_hot[torch.arange(classes_real.size(0)), classes_real] = 1
            embedded_class = self.embed_class(one_hot).view(-1, self.image_size, self.image_size)
            print(x.size())
            embedded_input = self.embed_data(x)
            x = torch.cat((embedded_input, embedded_class), dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def get_loss(self, recon_x, x, mu, log_var):
        bce = F.binary_cross_entropy(recon_x.view(-1, 32, 32), x.view(-1, 32, 32), reduction="sum")
        # mse = torch.nn.MSELoss()(recon_x.view(x.size()), x)
        kld = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
        # info-vae part
        # ce = F.nll_loss(input=classes_pred, target=classes_real)
        return bce, kld * self.beta


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

    def discriminate(self, x):
        return self.discriminator_head(x).view(-1, 1)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def log_gaussian(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)
