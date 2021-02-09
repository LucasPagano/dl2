from torchvision import transforms
import numpy as np
import torch


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
