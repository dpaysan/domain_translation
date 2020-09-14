import torch
from torch import Tensor
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


def compute_kld_multivariate_gaussians(mu:Tensor, logvar:Tensor, mu_prior=None, logvar_prior:Tensor=None)->Tensor:
    if mu_prior is None:
        mu_prior = torch.zeros_like(mu)
    if logvar_prior is None:
        logvar_prior = torch.ones_like(logvar)

    #KLloss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = 0.5 * torch.sum(logvar_prior - logvar - 1 + (mu-mu_prior).pow(2)/logvar_prior.exp() +logvar.exp()/logvar_prior.exp())
    return kld


def compute_kld_categoricals(ps:Tensor, qs:Tensor=None)->Tensor:
    if qs is None:
        qs = torch.ones_like(ps).div(ps.size()[1])

    kld = torch.sum(ps * torch.log(ps) - qs * torch.log(qs))
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


def max_discrepancy_loss(inputs: Tensor, outputs: Tensor) -> Tensor:
    n_samples, _ = inputs.size()
    loss = nn.L1Loss()(inputs, outputs) * (n_samples - 1) * n_samples
    discrepancy = 0
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                discrepancy += torch.mean(torch.abs(inputs[i] - outputs[j]))
    if discrepancy == 0:
        return loss * discrepancy
    else:
        return loss / discrepancy
