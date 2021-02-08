from torchvision import transforms


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
