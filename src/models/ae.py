from abc import abstractmethod, ABC
from typing import Any, List, Tuple

from torch import nn, Tensor


class BaseAE(nn.Module):
    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encode(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaAE(BaseAE, ABC):
    def __init__(
        self,
        in_dims: int = 2613,
        latent_dim: int = 128,
        hidden_dims: List = None,
        batchnorm_latent: bool = False,
        lrelu_slope: float = 0.2,
    ):
        super(VanillaAE, self).__init__()
        self.in_dims = in_dims
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.batchnorm_latent = batchnorm_latent
        self.lrelu_slope = lrelu_slope
        self.model_type = "AE"

        # Build encoder model
        if len(self.hidden_dims) > 0:
            encoder_modules = [
            nn.Sequential(
                nn.Linear(self.in_dims, self.hidden_dims[0]),
                #nn.BatchNorm1d(self.hidden_dims[0]),
                nn.LeakyReLU(self.lrelu_slope),
                nn.BatchNorm1d(self.hidden_dims[0]),
            )
        ]


        for i in range(1, len(hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    #nn.BatchNorm1d(self.hidden_dims[i]),
                    nn.LeakyReLU(self.lrelu_slope),
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
                #nn.BatchNorm1d(self.hidden_dims[-1]),
                nn.LeakyReLU(self.lrelu_slope),
                nn.BatchNorm1d(self.hidden_dims[-1]),
            )
        ]
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                    #nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                    nn.LeakyReLU(self.lrelu_slope),
                    nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                )
            )

        decoder_modules.append(nn.Linear(self.hidden_dims[0], self.in_dims))

        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor) -> Tensor:
        latents = self.encoder(input=input)
        return latents

    def decode(self, input: Tensor) -> Any:
        output = self.decoder(input=input)
        return output

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        latents = self.encode(input=inputs)
        output = self.decode(latents)
        return output, latents
