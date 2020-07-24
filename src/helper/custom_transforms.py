import torch
from torch import Tensor


class UnNormalize(object):
    """ Adapated from `here<https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821>`_."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(input, self.mean, self.std):
            t.mul_(s).add_(m)
        return Tensor
