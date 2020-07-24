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


def kl_divergence(mu: Tensor, logsigma: Tensor) -> Tensor:
    kld = 1 + logsigma * 2 - mu.square() - logsigma.mul_(2).exp()
    kld = torch.sum(kld) * -0.5
    return kld


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