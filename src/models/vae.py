from abc import abstractmethod, ABC
from typing import List, Any, Tuple

import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.autograd import Variable
from torch.nn import Module

# from src.models.custom_networks import MixtureComponentInferenceNetwork
from src.functions.loss_functions import compute_kld_multivariate_gaussians
from src.helper.custom_layers import SparseLinear
from src.utils.torch.general import get_device


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recon_loss_module = None
        self.model_base_type = "vae"

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def reparameterize(self, **kwargs) -> Tensor:
        raise NotImplementedError

    def sample(self, num_samples: int, device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    def loss_function(
            self, inputs: Tensor, recons: Tensor, mu: Tensor, logvar: Tensor
    ) -> dict:
        recon_loss = self.recon_loss_module(inputs, recons)
        kld_loss = compute_kld_multivariate_gaussians(mu=mu, logvar=logvar)
        loss_dict = {"recon_loss": recon_loss, "kld_loss": kld_loss}
        return loss_dict


class VanillaConvVAE(BaseVAE, ABC):
    def __init__(
            self,
            input_channels: int = 1,
            latent_dim: int = 128,
            hidden_dims: List[int] = [128, 256, 512, 1024, 1024],
            lrelu_slope: int = 0.2,
            batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lrelu_slope = lrelu_slope
        self.batchnorm = batchnorm
        self.updated = False
        # self.model_type = "VAE"
        self.n_latent_spaces = 1

        # encoder
        encoder_modules = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.hidden_dims[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                # nn.LeakyReLU(self.lrelu_slope, inplace=True),
                nn.PReLU(),
            )
        ]

        for i in range(1, len(self.hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.hidden_dims[i - 1],
                        out_channels=self.hidden_dims[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.hidden_dims[i]),
                    # nn.LeakyReLU(self.lrelu_slope, inplace=True),
                    nn.PReLU(),
                )
            )

        # Output of encoder are of shape 1024x2x2
        self.encoder = nn.Sequential(*encoder_modules)
        self.device = get_device()

        if self.batchnorm:
            self.mu_fc = nn.Sequential(
                nn.Linear(hidden_dims[-1] * 2 * 2, self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
            )
        else:
            self.mu_fc = nn.Linear(hidden_dims[-1] * 2 * 2, self.latent_dim)
        self.logvar_fc = nn.Linear(hidden_dims[-1] * 2 * 2, self.latent_dim)

        self.d1 = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dims[-1] * 2 * 2), nn.ReLU(inplace=True)
        )

        # decoder
        decoder_modules = []
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[-1 - i],
                        out_channels=hidden_dims[-2 - i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dims[-2 - i]),
                    # nn.LeakyReLU(self.lrelu_slope, inplace=True),
                    nn.PReLU(),
                )
            )
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[0],
                    out_channels=self.in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.Sigmoid(),
            )
        )

        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(input)
        h = h.view(-1, self.hidden_dims[-1] * 2 * 2)
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)

        return mu, logvar

    def decode(self, input: Tensor) -> Tensor:
        h = self.d1(input)
        h = h.view(-1, self.hidden_dims[-1], 2, 2)
        output = self.decoder(h)
        return output

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = logvar.mul(0.5).exp()
        eps = Variable(torch.FloatTensor(std.size()).normal_().to(self.device))
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, input: Tensor) -> dict:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        recons = self.decode(z)
        output = {"recons": recons, "latents": z, "mu": mu, "logvar": logvar}
        return output

    def get_latent_representation(self, input: Tensor) -> Tensor:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return z

    def generate(self, z: Tensor, **kwargs) -> Tensor:
        output = self.decode(z)
        return output

    def sample(self, num_samples: int, device: torch.device, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def loss_function(
            self, inputs: Tensor, recons: Tensor, mu: Tensor, logvar: Tensor
    ) -> dict:
        return super().loss_function(inputs=inputs, recons=recons, mu=mu, logvar=logvar)


class VanillaVAE(BaseVAE, ABC):
    def __init__(
            self,
            input_dim: int = 7633,
            latent_dim: int = 128,
            hidden_dims: List = None,
            batchnorm: bool = False,
            lrelu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if hidden_dims is None:
            self.hidden_dims = [1024, 1024, 1024, 1024, 1024, 1024]
        else:
            self.hidden_dims = hidden_dims
        self.batchnorm = batchnorm
        self.updated = False
        self.lrelu_slope = lrelu_slope
        # self.model_type = "VAE"

        # encoder
        encoder_modules = [
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dims[0]),
                nn.BatchNorm1d(self.hidden_dims[0]),
                nn.PReLU()
            ),
            # nn.LeakyReLU(self.lrelu_slope),
        ]
        for i in range(1, len(self.hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    #nn.LeakyReLU(self.lrelu_slope),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[i]),
                )
            )

        self.encoder = nn.Sequential(*encoder_modules)
        if self.batchnorm:
            self.mu_fc = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
            )
        else:
            self.mu_fc = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.logvar_fc = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # decoder
        decoder_modules = [
            nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dims[-1]))
        ]
        for i in range(0, len(self.hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                    #nn.LeakyReLU(self.lrelu_slope),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                )
            )

        decoder_modules.append(nn.Linear(self.hidden_dims[0], self.input_dim))
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(input)
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)
        return mu, logvar

    def decode(self, input: Tensor) -> Tensor:
        output = self.decoder(input)
        return output

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = logvar.mul(0.5).exp()
        eps = Variable(torch.FloatTensor(std.size()).normal_().to(mu.device))
        z = eps * std + mu
        return z

    def forward(self, input: Tensor) -> dict:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        recons = self.decode(z)
        output = {"recons": recons, "latents": z, "mu": mu, "logvar": logvar}
        return output

    def get_latent_representation(self, input: Tensor) -> Tensor:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return z

    def generate(self, z: Tensor, **kwargs) -> Tensor:
        output = self.decode(z)
        return output

    def sample(self, num_samples: int, device: torch.device, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def loss_function(
            self, inputs: Tensor, recons: Tensor, mu: Tensor, logvar: Tensor
    ) -> dict:
        return super().loss_function(inputs=inputs, recons=recons, mu=mu, logvar=logvar)


class GeneSetVAE(BaseVAE, ABC):
    def __init__(
            self,
            input_dim: int = 7633,
            latent_dim: int = 128,
            hidden_dims: List = None,
            batchnorm: bool = False,
            geneset_adjacencies: Tensor = None,
            geneset_adjacencies_file=None,
            lrelu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if hidden_dims is None:
            self.hidden_dims = [1024, 1024, 1024, 1024, 1024, 1024]
        else:
            self.hidden_dims = hidden_dims
        self.batchnorm = batchnorm
        self.updated = False
        self.lrelu_slope = lrelu_slope
        if geneset_adjacencies is not None:
            self.geneset_adjacencies = nn.Parameter(
                geneset_adjacencies, requires_grad=False
            )
        elif geneset_adjacencies_file is not None:
            geneset_adjacencies = pd.read_csv(geneset_adjacencies_file, index_col=0)
            self.geneset_adjacencies = nn.Parameter(
                torch.from_numpy(np.array(geneset_adjacencies)), requires_grad=False
            )
        else:
            raise RuntimeError(
                "Adjacency matrix must be given as a Tensor as the geneset_adjacencies parameter or a path to a .csv \
                file as the geneset_adjacencies_file parameter."
            )
        self.n_genesets = self.geneset_adjacencies.size()[1]

        # encoder
        self.geneset_encoding_layer = SparseLinear(
            self.input_dim, self.n_genesets, self.geneset_adjacencies,
        )
        self.geneset_encoder = self.geneset_encoding_layer

        encoder_modules = [nn.BatchNorm1d(self.n_genesets), nn.PReLU(),]
        if len(hidden_dims) > 0:
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.n_genesets, self.hidden_dims[0]),
                    #nn.LeakyReLU(self.lrelu_slope),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                    nn.PReLU(),
                )
            )
            for i in range(1, len(hidden_dims)):
                encoder_modules.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                        #nn.LeakyReLU(self.lrelu_slope),
                        nn.BatchNorm1d(self.hidden_dims[i]),
                        nn.PReLU(),
                    )
                )

        self.encoder = nn.Sequential(*encoder_modules)

        if self.batchnorm:
            self.mu_fc = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
            )
        else:
            self.mu_fc = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.logvar_fc = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Build decoder model
        if len(hidden_dims) > 0:
            decoder_modules = [
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dims[-1]),
                    #nn.LeakyReLU(self.lrelu_slope),
                    nn.BatchNorm1d(self.hidden_dims[-1]),
                    nn.PReLU(),
                )
            ]
            for i in range(len(hidden_dims) - 1):
                decoder_modules.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                        #nn.LeakyReLU(self.lrelu_slope),
                        nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                        nn.PReLU(),
                    )
                )
            decoder_modules.append(
                    nn.Linear(self.hidden_dims[0], self.n_genesets)
            )
        else:
            decoder_modules = [
                nn.Linear(self.latent_dim, self.n_genesets)
            ]

        self.decoder = nn.Sequential(*decoder_modules)

        self.geneset_decoder = nn.Sequential(
            #nn.LeakyReLU(self.lrelu_slope),
            nn.BatchNorm1d(self.n_genesets),
            nn.PReLU(),
            SparseLinear(self.n_genesets, self.input_dim, self.geneset_adjacencies),
        )

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        geneset_activities = self.geneset_encoder(input)
        h = self.encoder(geneset_activities)
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)
        return mu, logvar, geneset_activities

    def decode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        decoded_geneset_activities = self.decoder(input)
        recons = self.geneset_decoder(decoded_geneset_activities)
        return recons, decoded_geneset_activities

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = logvar.mul(0.5).exp()
        eps = Variable(torch.FloatTensor(std.size()).normal_().to(mu.device))
        latents = eps * std + mu
        return latents

    def forward(self, input: Tensor) -> dict:
        mu, logvar, geneset_activities = self.encode(input)
        latents = self.reparameterize(mu, logvar)
        recons, decoded_geneset_activities = self.decode(latents)
        output = {"recons": recons, "latents": latents, "mu": mu, "logvar": logvar,
                  "geneset_activities":geneset_activities, "decoded_geneset_activities":decoded_geneset_activities}
        return output

    def get_latent_representation(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        mu, logvar, geneset_activities = self.encode(input)
        latents = self.reparameterize(mu, logvar)
        return latents, geneset_activities

    def generate(self, z: Tensor, **kwargs) -> Tensor:
        output, _ = self.decode(z)
        return output

    def sample(self, num_samples: int, device: torch.device, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples, _ = self.decode(z)
        return samples

    def loss_function(
            self, inputs: Tensor, recons: Tensor, mu: Tensor, logvar: Tensor
    ) -> dict:
        return super().loss_function(inputs=inputs, recons=recons, mu=mu, logvar=logvar)


# class GaussianMixtureBaseVAE(BaseVAE, ABC):
#     def __init__(
#             self,
#             input_dim: int = 7633,
#             latent_dim: int = 128,
#             hidden_dims: List = None,
#             batchnorm: bool = False,
#             lrelu_slope: float = 0.2,
#             reconstruction_distribution: str = "gaussian",
#             n_mixture_components: int = 1,
#     ):
#         super(BaseVAE, self).__init__()
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.hidden_dims = hidden_dims
#         self.batchnorm = batchnorm
#         self.lrelu_slope = lrelu_slope
#         self.reconstruction_distribution = reconstruction_distribution
#         self.n_mixture_components = n_mixture_components
#
#         self.encoder = None
#         self.decoder = None
#
#     def encode(self, input: Tensor) -> List[Tensor]:
#         raise NotImplementedError
#
#     def decode(self, input: Tensor) -> Any:
#         raise NotImplementedError
#
#     def reparameterize(self):
#         raise NotImplementedError
#
#     def forward(self, *inputs: Tensor) -> Tensor:
#         raise NotImplementedError
#
#     def generate(self, x: Tensor, **kwargs) -> Tensor:
#         raise NotImplementedError
#
#     def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
#         raise NotImplementedError
#
#
# class GaussianMixtureEncoder(Module, ABC):
#     def __init__(
#             self,
#             input_dim: int = 7633,
#             latent_dim: int = 128,
#             hidden_dims_qyx: List = None,
#             hidden_dims_qzyx: List = None,
#             batchnorm: bool = False,
#             n_mixture_components: int = 1,
#     ):
#         super().__init__()
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.hidden_dims_qyx = hidden_dims_qyx
#         self.hidden_dims_qzyx = hidden_dims_qzyx
#         self.batchnorm = batchnorm
#         self.n_mixture_components = n_mixture_components
#
#         # q(y|x)
#         qyx_modules = [
#             nn.Sequential(
#                 nn.Linear(self.in_dims, self.hidden_dims_qyx[0]),
#                 nn.BatchNorm1d(self.hidden_dims_qyx[0]),
#             ),
#             nn.PReLU(),
#         ]
#         for i in range(1, len(self.hidden_dims_qyx)):
#             qyx_modules.append(
#                 nn.Sequential(
#                     nn.Linear(self.hidden_dims_qyx[i - 1], self.hidden_dims_qyx[i]),
#                     nn.BatchNorm1d(self.hidden_dims_qyx[i]),
#                     nn.PReLU(),
#                 )
#             )
#
#         self.qyx_network = MixtureComponentInferenceNetwork(nn.Sequential(*qyx_modules),
#                                                             GumbelSoftMax(self.hidden_dims_qyx[-1],
#                                                                           self.n_mixture_components))
#
#         qzyx_modules = [
#             nn.Sequential(
#                 nn.Linear(self.in_dims + self.n_mixture_components, self.hidden_dims_qzyx[0]),
#                 nn.BatchNorm1d(self.hidden_dims_qzyx[0]),
#             ),
#             nn.PReLU(),
#         ]
#         for i in range(1, len(self.hidden_dims_qzyx)):
#             qzyx_modules.append(
#                 nn.Sequential(
#                     nn.Linear(self.hidden_dims_qzyx[i - 1], self.hidden_dims_qzyx[i]),
#                     nn.BatchNorm1d(self.hidden_dims_qzyx[i]),
#                     nn.PReLU(),
#                 )
#             )
#         qzyx_modules.append(Gaussian(self.hidden_dims_qzyx[-1], self.latent_dim))
#
#         self.qzyx_network = nn.Sequential(*qzyx_modules)
#
#     def forward(self, *inputs: Tensor, temperature: float = 1.0, hard: float = 0) -> dict:
#         logits, prob, y = self.qyx_network(inputs, temperature, hard)
#
#         xy = torch.cat([inputs, y], dim=1)
#         mu, var, z = self.qzyx_network(xy)
#
#         output = {'mu': mu, 'var': var, 'latents': z, 'logits': logits, 'responsibility': prob, 'component_label': y}
#
#         return output

