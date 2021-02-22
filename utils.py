from torchvision import transforms
import numpy as np
import torch


def get_optimizer(conf, nn):
    if conf.optimizer == "Adam":
        opt = torch.optim.Adam(nn.parameters(), lr=conf.lr)
    elif conf.optimizer == "RMSprop":
        opt = torch.optim.RMSprop(nn.parameters(), lr=conf.lr)
    elif conf.optimizer == "SGD":
        opt = torch.optim.SGD(nn.parameters(), lr=conf.lr)
    return opt


def get_unorm(mean, std):
    """
    Returns a transform to undo a normalization, useful for visualization
    :param mean:
    :param std:
    :return: pytorch transforms that undoes normalization
    """

    m_mean = [-i for i in mean]
    std_inv = [1 / i for i in std]
    mean_inv = [i * j for i, j in zip(m_mean, std_inv)]
    return transforms.Normalize(mean=mean_inv, std=std_inv)


def noise_sample(dis_c, con_c, noise, bs):
    """Sample a noise vector as a concatenation of noise, discrete and continuous latent bits"""
    idx = np.random.randint(10, size=bs)
    one_hot = np.zeros((bs, 10))
    one_hot[range(bs), idx] = 1.0

    dis_c.data.copy_(torch.Tensor(one_hot))
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
    return z, idx


def get_latent_steps(nb_epochs, latent_size, nb_cuts_latent):
    """Return a dictionary matching the epoch to the latent size to be trained"""
    assert nb_cuts_latent <= latent_size

    step_latent = latent_size // nb_cuts_latent
    first_latent = step_latent + latent_size % nb_cuts_latent

    step_epoch = nb_epochs // nb_cuts_latent
    first_epoch = step_epoch + nb_epochs % nb_cuts_latent

    size_latent = first_latent
    epoch_stop = first_epoch
    epoch_dict = {}
    for e in range(nb_epochs):
        if e < epoch_stop:
            epoch_dict[e] = size_latent
        else:
            size_latent += step_latent
            epoch_stop += step_epoch
            epoch_dict[e] = size_latent

    return epoch_dict


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    print(get_latent_steps(15, 5, 3))
    print(get_latent_steps(15, 10, 3))
    print(get_latent_steps(8, 2, 2))
    print(get_latent_steps(8, 50, 4))
