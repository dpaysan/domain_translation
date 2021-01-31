from abc import abstractmethod, ABC
from typing import Any, List, Tuple
import torch
from torch import nn, Tensor
import numpy as np
import pandas as pd

from src.helper.custom_layers import SparseLinear
from src.utils.torch.general import get_device


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
        self.n_latent_spaces = 1

        # Build encoder model
        if len(self.hidden_dims) > 0:
            encoder_modules = [
                nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dims[0]),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                )
            ]

        for i in range(1, len(hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    # nn.LeakyReLU(self.lrelu_slope),
                    nn.ReLU(),
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
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dims[-1]),
            )
        ]
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                    nn.ReLU(),
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


class VanillaConvAE(BaseAE, ABC):
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
        self.n_latent_spaces = 1

        # Build encoder
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
                nn.ReLU(),
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
                    nn.ReLU(),
                )
            )
        self.encoder = nn.Sequential(*encoder_modules)

        # Output of encoder are of shape 1024x4x4
        self.device = get_device()

        if self.batchnorm:
            self.latent_mapper = nn.Sequential(
                nn.Linear(hidden_dims[-1] * 2 * 2, self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
            )
        else:
            self.latent_mapper = nn.Linear(hidden_dims[-1] * 2 * 2, self.latent_dim)

        self.inv_latent_mapper = nn.Sequential(
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
                    nn.ReLU(),
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

    def encode(self, input: Tensor) -> Tensor:
        features = self.encoder(input=input)
        features = features.view(features.size(0), -1)
        latents = self.latent_mapper(input=features)
        return latents

    def decode(self, input: Tensor) -> Any:
        latent_features = self.inv_latent_mapper(input)
        latent_features = latent_features.view(
            latent_features.size(0), self.hidden_dims[-1], 2, 2
        )
        output = self.decoder(input=latent_features)
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
        self.n_latent_spaces = 2

        # Build encoder model
        if len(self.hidden_dims) > 0:
            encoder_modules = [
                nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dims[0]),
                    #nn.LeakyReLU(0.2),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                )
            ]

        for i in range(1, len(hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    #nn.LeakyReLU(0.2),
                    nn.ReLU(),
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
                #nn.LeakyReLU(0.2),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dims[-1]),
            )
        ]
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                    #nn.LeakyReLU(0.2),
                    nn.ReLU(),
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


class GeneSetAE(BaseAE, ABC):
    def __init__(
        self,
        input_dim: int = 2613,
        latent_dim: int = 128,
        hidden_dims: List = [512,512,512, 512],
        batchnorm: bool = True,
        geneset_adjacencies: Tensor = None,
        geneset_adjacencies_file=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.batchnorm = batchnorm
        if geneset_adjacencies is not None:
            self.geneset_adjacencies = nn.Parameter(geneset_adjacencies, requires_grad=False)
        elif geneset_adjacencies_file is not None:
            geneset_adjacencies = pd.read_csv(geneset_adjacencies_file,
                index_col=0)
            self.geneset_adjacencies = nn.Parameter(torch.from_numpy(np.array(geneset_adjacencies)),
                                                    requires_grad=False)
        else:
            raise RuntimeError(
                "Adjacency matrix must be given as a Tensor as the geneset_adjacencies parameter or a path to a .csv \
                file as the geneset_adjacencies_file parameter."
            )
        self.n_latent_spaces = 1
        self.n_genesets = self.geneset_adjacencies.size()[1]

        self.geneset_encoding_layer = SparseLinear(
            self.input_dim, self.n_genesets, self.geneset_adjacencies,
        )
        self.geneset_encoder = nn.Sequential(self.geneset_encoding_layer, nn.ReLU(), nn.BatchNorm1d(self.n_genesets))
        if len(hidden_dims) > 0:
            encoder_modules = [
                nn.Sequential(
                    nn.Linear(self.n_genesets, self.hidden_dims[0]),
                    #nn.LeakyReLU(0.2),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                )
            ]
            for i in range(1, len(hidden_dims)):
                encoder_modules.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                        #nn.LeakyReLU(0.2),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.hidden_dims[i]),
                    )
                )
            encoder_modules.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))
        else:
            encoder_modules = [
                nn.Sequential(nn.Linear(self.n_genesets, self.latent_dim))
            ]

        if self.batchnorm:
            encoder_modules.append(nn.BatchNorm1d(self.latent_dim))

        self.encoder = nn.Sequential(*encoder_modules)

        # Build decoder model
        if len(hidden_dims) > 0:
            decoder_modules = [
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dims[-1]),
                    #nn.LeakyReLU(0.2),
                    nn.ReLU(),
                    #nn.BatchNorm1d(self.hidden_dims[-1]),
                )
            ]
            for i in range(len(hidden_dims) - 1):
                decoder_modules.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                        #nn.LeakyReLU(0.2),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                    )
                )
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[0], self.n_genesets), nn.ReLU(), nn.BatchNorm1d(self.n_genesets)
                )
            )
        else:
            decoder_modules = [
                nn.Sequential(nn.Linear(self.latent_dim, self.n_genesets), nn.ReLU(), nn.BatchNorm1d(self.n_genesets))
            ]

        self.geneset_decoder = nn.Sequential(
            SparseLinear(self.n_genesets, self.input_dim, self.geneset_adjacencies)
        )
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, inputs: Tensor) -> Any:
        geneset_activities = self.geneset_encoder(inputs)
        latents = self.encoder(geneset_activities)
        return latents

    def decode(self, latents: Tensor) -> Any:
        decoded_geneset_activities = self.decoder(latents)
        recons = self.geneset_decoder(decoded_geneset_activities)
        return recons

    def forward(self, inputs: Tensor) -> Any:
        latents = self.encode(inputs)
        recons = self.decode(latents)
        return {"recons": recons, "latents": latents}
