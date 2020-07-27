from abc import abstractmethod, ABC
from typing import List, Any, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.autograd import Variable

from src.utils.torch.general import get_device


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, num_samples: int, device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaConvVAE(BaseVAE, ABC):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: List = None,
        **kwargs
    ) -> None:
        super(BaseVAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        if not self.hidden_dims:
            self.hidden_dims = [128, 256, 512, 1024, 1024]

        # encoder
        encoder_modules = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.LeakyReLU(self.lrelu_slope, inplace=True),
            )
        ]

        for i in range(1, len(hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.hidden_dims[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.hidden_dims[i]),
                    nn.LeakyReLU(self.lrelu_slope, inplace=True),
                )
            )

        # Output of encoder are of shape 1024x2x2
        self.encoder = nn.Sequential(*encoder_modules)
        self.device = get_device()

        self.fc1 = nn.Linear(hidden_dims[-1] * 2 * 2, self.latent_dim)
        self.fc2 = nn.Linear(hidden_dims[-1] * 2 * 2, self.latent_dim)

        self.d1 = nn.Linear(self.latent_dim, hidden_dims[-1] * 2 * 2)

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
                    ),
                    nn.BatchNorm2d(hidden_dims[-2 - i]),
                    nn.ReLU(inplace=True),
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
                ),
                nn.Sigmoid(),
            )
        )

        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(input)
        h = h.view(-1, self.latent_dim)
        mu = self.fc1(h)
        logsigma = self.fc2(h)

        return mu, logsigma

    def decode(self, input: Tensor) -> Tensor:
        h = self.d1(input)
        h = h.view(-1, self.hidden_dims[-1], 2, 2)
        output = self.decoder(h)
        return output

    def reparameterize(self, mu: Tensor, logsigma: Tensor) -> Tensor:
        std = logsigma.mul(0.5).exp()
        eps = Variable(torch.FloatTensor(std.size()).normal_().to(self.device))
        z = eps * std + mu
        return z

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logsigma = self.encode(input)
        z = self.reparameterize(mu, logsigma)
        output = self.decode(z)
        return output, z, mu, logsigma

    def get_latent_representation(self, input: Tensor) -> Tensor:
        mu, logsigma = self.encode(input)
        z = self.reparameterize(mu, logsigma)
        return z

    def generate(self, z: Tensor, **kwargs) -> Tensor:
        output = self.decode(z)
        return output

    def sample(self, num_samples: int, device: torch.device, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples


class VanillaVAE(BaseVAE, ABC):
    def __init__(
        self,
        in_dims: int = 1,
        latent_dim: int = 128,
        hidden_dims: List = None,
        **kwargs
    ) -> None:
        super(BaseVAE, self).__init__()
        self.in_dims = in_dims
        self.latent_dim = latent_dim
        if hidden_dims is None:
            self.hidden_dims = [1024, 1024, 1024, 1024, 1024, 1024]
        else:
            self.hidden_dims = hidden_dims

        # encoder
        encoder_modules = [
            nn.Sequential(
                nn.Linear(self.in_dims, self.hidden_dims[0]),
                nn.BatchNorm1d(self.hidden_dims[0]),
            ),
            nn.ReLU(),
        ]
        for i in range(1, len(self.hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    nn.BatchNorm1d(self.hidden_dims[i]),
                    nn.ReLU(),
                )
            )

        self.encoder = nn.Sequential(*encoder_modules)
        self.mu_fc = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.logsgima_fc = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # decoder
        decoder_modules = [
            nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dims[-1]))
        ]
        for i in range(0, len(self.hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                    nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                    nn.ReLU(),
                )
            )

        decoder_modules.append(nn.Linear(self.hidden_dims[0], self.in_dims))
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(input)
        mu = self.mu_fc(h)
        logsigma = self.logsgima_fc(h)

        return mu, logsigma

    def decode(self, input: Tensor) -> Tensor:
        output = self.decoder(input)
        return output

    def reparameterize(self, mu: Tensor, logsigma: Tensor) -> Tensor:
        std = logsigma.mul(0.5).exp()
        eps = Variable(torch.FloatTensor(std.size()).normal_().to(self.device))
        z = eps * std + mu
        return z

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logsigma = self.encode(input)
        z = self.reparameterize(mu, logsigma)
        output = self.decode(z)
        return output, z, mu, logsigma

    def get_latent_representation(self, input: Tensor) -> Tensor:
        mu, logsigma = self.encode(input)
        z = self.reparameterize(mu, logsigma)
        return z

    def generate(self, z: Tensor, **kwargs) -> Tensor:
        output = self.decode(z)
        return output

    def sample(self, num_samples: int, device: torch.device, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples
