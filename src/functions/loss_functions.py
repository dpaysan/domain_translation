import torch
from torch import Tensor
import numpy as np
from torch import nn


def p_norm_loss(
    input: Tensor, target: Tensor, p: int, reduction: str = "mean"
) -> Tensor:
    norm = torch.norm(input - target, p=p)
    if reduction == "mean":
        return norm / len(input)
    elif reduction not in ["none", "sum"]:
        raise NotImplementedError


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    # kld = torch.mean(
    #   -0.5 * torch.sum(1 + 2 * log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    # )
    kl_loss = 1 + logvar * 2 - mu.square() - logvar.mul_(2).exp()
    kl_loss = torch.sum(kl_loss) * -0.5
    return kl_loss


def accuracy_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    r""" Value function assessing the prediction accuracy of a probabilistic classifier.

    Parameters
    ----------
    input
    target
    reduction

    Returns
    -------

    """
    n_classes = target.size()[1]
    hot_one_encoded_input = np.zeros(input.size(), n_classes)
    hot_one_encoded_input[np.arange(input.size()), input] = 1
    # hot_one_encoded_input = torch.from_numpy(hot_one_encoded_input)
    acc_loss = torch.sum(torch.log(target[hot_one_encoded_input]))
    if reduction == "mean":
        return acc_loss / len(input)
    elif reduction not in ["none", "sum"]:
        raise NotImplementedError
    return acc_loss


class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

        self.loss_function = kl_divergence

    def forward(self, mu, logvar):
        return self.loss_function(mu, logvar)


def vae_loss(inputs: Tensor, outputs: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    batch_size, image_size, _ = inputs.size()
    recon_loss = nn.BCELoss()(inputs.view(batch_size, -1), outputs.view(batch_size, -1))
    recon_loss *= image_size * image_size
    kl_loss = 1 + log_var * 2 - mu.square() - log_var.mul_(2).exp()
    kl_loss = torch.sum(kl_loss) * -0.5
    vae_loss = torch.mean(recon_loss + kl_loss)
    return vae_loss


class BCELoss_transformed(nn.Module):
    def __init__(self):
        super(BCELoss_transformed, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs: Tensor, recons: Tensor) -> Tensor:
        batch_size, input_size, _ = inputs.size()
        inputs = inputs.view(batch_size, -1)
        recons = recons.view(batch_size, -1)
        recon_loss = self.bce_loss(inputs, recons)
        recon_loss *= input_size * input_size
        return recon_loss
