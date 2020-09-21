from abc import abstractmethod, ABC
from typing import Any, List, Tuple

import torch
from torch import nn, Tensor


class BaseAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recon_loss_module = None
        self.model_base_type = "ae"

    def encode(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        pass

    def loss_function(self, inputs: Tensor, recons: Tensor) -> dict:
        recon_loss = self.recon_loss_module(inputs, recons)
        loss_dict = {"recon_loss": recon_loss}
        return loss_dict


class VanillaAE(BaseAE, ABC):
    def __init__(
        self,
        input_dim: int = 2613,
        latent_dim: int = 128,
        hidden_dims: List = None,
        batchnorm_latent: bool = False,
        lrelu_slope: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.batchnorm_latent = batchnorm_latent
        self.lrelu_slope = lrelu_slope
        # self.model_type = "AE"
        self.n_latent_spaces = 1

        # Build encoder model
        if len(self.hidden_dims) > 0:
            encoder_modules = [
                nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dims[0]),
                    # nn.LeakyReLU(self.lrelu_slope),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                )
            ]

        for i in range(1, len(hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    # nn.LeakyReLU(self.lrelu_slope),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[i]),
                )
            )
        if self.batchnorm_latent:
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1], self.latent_dim),
                    nn.BatchNorm1d(self.latent_dim),
                )
            )
        else:
            encoder_modules.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))

        self.encoder = nn.Sequential(*encoder_modules)

        # Build decoder model
        decoder_modules = [
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dims[-1]),
                # nn.LeakyReLU(self.lrelu_slope),
                nn.PReLU(),
                nn.BatchNorm1d(self.hidden_dims[-1]),
            )
        ]
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                    # nn.LeakyReLU(self.lrelu_slope),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                )
            )

        decoder_modules.append(nn.Linear(self.hidden_dims[0], self.input_dim))

        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor) -> Tensor:
        latents = self.encoder(input=input)
        return latents

    def decode(self, input: Tensor) -> Any:
        output = self.decoder(input=input)
        return output

    def forward(self, inputs: Tensor) -> dict:
        latents = self.encode(input=inputs)
        recons = self.decode(latents)
        output = {"recons": recons, "latents": latents}
        return output


class TwoLatentSpaceAE(BaseAE, ABC):
    def __init__(
        self,
        input_dim: int = 2613,
        latent_dim_1: int = 128,
        latent_dim_2: int = 64,
        hidden_dims: List = None,
        batchnorm_latent: bool = False,
        lrelu_slope: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        self.hidden_dims = hidden_dims
        self.batchnorm_latent = batchnorm_latent
        self.lrelu_slope = lrelu_slope
        # self.model_type = "AE"
        self.n_latent_spaces = 2

        # Build encoder model
        if len(self.hidden_dims) > 0:
            encoder_modules = [
                nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dims[0]),
                    # nn.BatchNorm1d(self.hidden_dims[0]),
                    # nn.LeakyReLU(self.lrelu_slope),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                )
            ]

        for i in range(1, len(hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    # nn.LeakyReLU(self.lrelu_slope),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[i]),
                )
            )

        self.encoder = nn.Sequential(*encoder_modules)

        self.l1 = nn.Linear(self.hidden_dims[-1], self.latent_dim_1)
        self.l2 = nn.Linear(self.hidden_dims[-1], self.latent_dim_2)

        # Build decoder model
        decoder_modules = [
            nn.Sequential(
                nn.Linear(self.latent_dim_1 + latent_dim_2, self.hidden_dims[-1]),
                # nn.LeakyReLU(self.lrelu_slope),
                nn.PReLU(),
                nn.BatchNorm1d(self.hidden_dims[-1]),
            )
        ]
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                    # nn.LeakyReLU(self.lrelu_slope),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                )
            )

        decoder_modules.append(nn.Linear(self.hidden_dims[0], self.input_dim))

        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(input=input)
        latents_1 = self.l1(h)
        latents_2 = self.l2(h)
        return latents_1, latents_2

    def decode(self, input: Tensor) -> Any:
        output = self.decoder(input=input)
        return output

    def forward(self, inputs: Tensor) -> dict:
        latents_1, latents_2 = self.encode(input=inputs)
        latents = torch.cat([latents_1, latents_2], dim=1)
        recons = self.decode(latents)
        output = {"recons": recons, "latents": latents_1, "unshared_latents": latents_2}
        return output

    def loss_function(self, inputs: Tensor, recons: Tensor) -> dict:
        return super().loss_function(inputs=inputs, recons=recons)
