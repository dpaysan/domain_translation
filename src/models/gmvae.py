# The implementation is inspired by the one available at https://github.com/RuiShu/vae-clustering/ .
from abc import ABC
from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, CrossEntropyLoss
from torch.autograd import Variable

from src.functions.loss_functions import compute_kld_multivariate_gaussians, compute_kld_categoricals
from src.utils.torch.general import get_activation_module


class QYXNetwork(Module, ABC):
    """ Realizes q(y|x) for the GMVAE"""

    def __init__(
        self, input_dim: int, hidden_dims: List, n_components: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_components = n_components

        modules = [
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dims[0]),
                nn.BatchNorm1d(self.hidden_dims[0]),
                nn.PReLU(),
            )
        ]

        for i in range(1, len(self.hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    nn.BatchNorm1d(self.hidden_dims[i]),
                    nn.PReLU(),
                )
            )

        modules.append(nn.Linear(self.hidden_dims[-1], self.n_components))

        self.network = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.network(x)
        probs = torch.softmax(logits, dim=1)
        y_s = torch.argmax(probs, dim=1)
        return logits, probs, y_s


class QZYXNetwork(Module, ABC):
    """ Realizes q(z|y,x) for the GMVAE"""

    def __init__(
        self, input_dim: int, hidden_dims: List, n_components: int, latent_dim: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_components
        self.latent_dim = latent_dim

        modules = [
            nn.Sequential(
                nn.Linear(self.input_dim + self.n_classes, self.hidden_dims[0]),
                nn.BatchNorm1d(self.hidden_dims[0]),
                nn.PReLU(),
            )
        ]

        for i in range(1, len(self.hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    nn.BatchNorm1d(self.hidden_dims[i]),
                    nn.PReLU(),
                )
            )

        self.network = nn.Sequential(*modules)
        self.mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.logsigma = nn.Linear(self.hidden_dims[-1], self.latent_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        xy = torch.cat([x, y], dim=1)
        h = self.network(xy)
        mu = self.mu(h)
        logsigma = self.logsigma(h)
        z = self.reparameterize(mu, logsigma)
        return z, mu, logsigma

    def reparameterize(self, mu: Tensor, sigma: Tensor):
        std = sigma.mul(0.5).exp()
        eps = Variable(torch.FloatTensor(std.size()).normal_().to(mu.device))
        z = eps * std + mu
        return z


class PZYNetwork(Module, ABC):
    """ Prior network for the class-conditional Gaussians"""

    def __init__(self, n_components: int, latent_dim: int):
        super().__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim

        self.mu = nn.Embedding(n_components, latent_dim)
        self.logvar = nn.Embedding(n_components, latent_dim)

    def forward(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        #y_onehot = torch.zeros(y.size()[0], self.n_components)
        #y_onehot = y_onehot.scatter(1,y,1)
        mu = self.mu(y)
        logvar = self.logvar(y)

        return mu, logvar


class QXZNetwork(Module, ABC):
    """ Realizes p(x|z) for the GMVAE"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List,
        latent_dim: int,
        output_activation: str = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        modules = [
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dims[-1]),
                nn.BatchNorm1d(self.hidden_dims[-1]),
                nn.PReLU(),
            )
        ]
        for i in range(0, len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                    nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                    nn.PReLU(),
                )
            )

        modules.append(nn.Linear(self.hidden_dims[0], self.input_dim))
        if output_activation is not None:
            modules.append(get_activation_module(output_activation))
        self.network = nn.Sequential(*modules)

    def forward(self, z: Tensor) -> Tensor:
        x_rec = self.network(z)
        return x_rec


class GaussianMixtureVAE(Module, ABC):
    def __init__(
        self,
        input_dim,
        hidden_dims:List,
        latent_dim: int,
        n_components: int,
        hidden_dims_qyx: List=None,
        hidden_dims_qzyx: List=None,
        hidden_dims_qxz: List=None,
    ):
        super().__init__()
        self.input_dim = input_dim

        if hidden_dims_qyx is not None:
            self.hidden_dims_qyx = hidden_dims_qyx
        else:
            self.hidden_dims_qyx = hidden_dims

        if hidden_dims_qzyx is not None:
            self.hidden_dims_qzyx = hidden_dims_qzyx
        else:
            self.hidden_dims_qzyx = hidden_dims

        if hidden_dims_qxz is not None:
            self.hidden_dims_qxz = hidden_dims_qxz
        else:
            self.hidden_dims_qxz = hidden_dims

        self.latent_dim = latent_dim
        self.n_components = n_components

        self.recon_loss_module = None
        self. model_base_type = 'gmvae'

        # Todo make that loss definition variable
        self.component_supervision_loss_module = CrossEntropyLoss()

        self.qyx_network = QYXNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims_qyx,
            n_components=self.n_components,
        )
        self.qzyx_network = QZYXNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims_qzyx,
            n_components=self.n_components,
            latent_dim=self.latent_dim,
        )
        self.qxz_network = QXZNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims_qxz,
            latent_dim=self.latent_dim,
        )

        self.pzy_network = PZYNetwork(n_components=n_components, latent_dim=latent_dim)

    def forward(self, x: Tensor) -> dict:
        logits, probs, y_s = self.qyx_network(x)
        z, mu, logvar = self.qzyx_network(x, logits)

        mu_prior, logvar_prior = self.pzy_network(y_s)
        recons = self.qxz_network(z)

        output = {
            "recons": recons,
            "latents": z,
            "mu": mu,
            "logvar": logvar,
            "logits": logits,
            "probs": probs,
            "component_labels": y_s,
            "mu_component_prior": mu_prior,
            "logvar_component_prior": logvar_prior,
        }

        return output

    def loss_function(self, inputs:Tensor, recons:Tensor, mu:Tensor, logvar:Tensor, mu_prior:Tensor,
                      logvar_prior:Tensor, y_probs:Tensor, y_logits:Tensor=None, y_true:Tensor=None) -> dict:

        recon_loss = self.recon_loss_module(inputs, recons)
        kld_loss = compute_kld_multivariate_gaussians(mu=mu, logvar=logvar, mu_prior=mu_prior, logvar_prior=logvar_prior)
        if y_true is None:
            component_prior_loss = compute_kld_categoricals(y_probs)
            component_supervision_loss = None
        else:
            component_prior_loss = None
            component_supervision_loss = self.component_supervision_loss_module(y_logits, y_true)
        loss_dict = {'recon_loss':recon_loss, 'kld_loss':kld_loss, 'component_prior_loss':component_prior_loss,
                     'component_supervision_loss':component_supervision_loss}

        return loss_dict
