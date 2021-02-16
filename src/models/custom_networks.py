from abc import ABC
from typing import Any, List
import torch

from torch import Tensor
from torch import nn

from src.models.ae import GeneSetAE, BaseAE, VanillaConvAE
import numpy as np
import pandas as pd


class GenesetEncoder(nn.Module, ABC):
    def __init__(self, adjacency_matrix: Tensor, hidden_dims: List):
        super().__init__()
        self.adjacency_matrix = adjacency_matrix
        self.hidden_dims = hidden_dims
        self.geneset_encoders = []

        self.input_dim, self.output_dim = self.adjacency_matrix.size()

        # Construct geneset modules
        for i in range(self.output_dim):
            geneset_size = torch.sum(self.adjacency_matrix[:, i]).data
            if len(self.hidden_dims) > 0:
                geneset_modules = [
                    nn.Sequential(
                        nn.Linear(geneset_size, self.hidden_dims[0]),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.hidden_dims[0]),
                    )
                ]
                for i in range(1, len(self.hidden_dims)):
                    geneset_modules.append(
                        nn.Sequential(
                            nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                            nn.ReLU(),
                            nn.BatchNorm1d(self.hidden_dims[i]),
                        )
                    )
                geneset_modules.append(
                    nn.Sequential(nn.Linear(self.hidden_dims[-1], 1), nn.ReLU())
                )
                self.geneset_encoders.append(nn.Sequential(*geneset_modules))
            else:
                self.geneset_encoders.append(
                    nn.Sequential(nn.Linear(geneset_size, 1), nn.ReLU())
                )

        self.geneset_encoders = nn.ModuleList(self.geneset_encoders)

    def forward(self, x: Tensor) -> Tensor:
        output = []
        for i in range(self.output_dim):
            inputs = x[:, self.adjacency_matrix[:, i] == 1]
            output.append(self.geneset_encoders[i](inputs))
        output = torch.cat(output, dim=1)
        return output


class GenesetDecoder(nn.Module, ABC):
    def __init__(self, adjacency_matrix: Tensor, hidden_dims: List):
        super().__init__()
        # Adjacency matrix will be of dimensions genesets x genes
        self.adjacency_matrix = nn.Parameter(
            adjacency_matrix.transpose(0, 1), requires_grad=False
        )
        self.hidden_dims = hidden_dims
        self.geneset_decoders = []

        self.input_dim, self.output_dim = self.adjacency_matrix.size()

        # Construct geneset modules
        for i in range(self.input_dim):
            if len(self.hidden_dims) > 0:
                gene_modules = [
                    nn.Sequential(
                        nn.Linear(1, self.hidden_dims[0]),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.hidden_dims[0]),
                    )
                ]
                for i in range(1, len(self.hidden_dims)):
                    gene_modules.append(
                        nn.Sequential(
                            nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                            nn.ReLU(),
                            nn.BatchNorm1d(self.hidden_dims[i]),
                        )
                    )
                gene_modules.append(
                    nn.Sequential(nn.Linear(self.hidden_dims[-1], self.output_dim))
                )
                self.geneset_decoders.append(nn.Sequential(*gene_modules))
            else:
                self.geneset_decoders.append(
                    nn.Sequential(nn.Linear(1, self.output_dim))
                )

        self.geneset_decoders = nn.ModuleList(self.geneset_decoders)

    def forward(self, x: Tensor) -> Tensor:
        # Todo: Think about generalizing the simplifying additive relationship of pathway activation for gene activity
        output = torch.zeros((x.size()[0], self.output_dim)).to(x.device)
        for i in range(self.input_dim):
            inputs = x[:, i].view(x.size()[0], 1)
            output += self.geneset_decoders[i](inputs) * self.adjacency_matrix[i, :]
        return output


class GeneSetAE_v2(BaseAE, ABC):
    def __init__(
        self,
        geneset_adjacencies: Tensor = None,
        geneset_adjacencies_file: str = None,
        hidden_dims: List = [64, 32, 16, 8],
        latent_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        if geneset_adjacencies is not None:
            self.geneset_adjacencies = nn.Parameter(
                geneset_adjacencies, requires_grad=False
            )
        elif geneset_adjacencies_file is not None:
            geneset_adjacencies = pd.read_csv(geneset_adjacencies_file, index_col=0)
            self.geneset_adjacencies = nn.Parameter(
                torch.from_numpy(np.array(geneset_adjacencies)), requires_grad=False
            )

        self.geneset_encoder = GenesetEncoder(
            adjacency_matrix=self.adjacency_matrix, hidden_dims=hidden_dims
        )
        self.geneset_decoder = GenesetDecoder(
            adjacency_matrix=self.adjacency_matrix, hidden_dims=hidden_dims[::-1]
        )
        self.latent_mapper = nn.Linear(self.geneset_encoder.output_dim, self.latent_dim)
        self.inv_latent_mapper = nn.Linear(
            self.latent_dim, self.geneset_decoder.input_dim
        )

    def forward(self, inputs: Tensor):
        geneset_activities = self.geneset_encoder(inputs)
        latents = self.latent_mapper(geneset_activities)
        inv_latents = self.inv_latent_mapper(latents)
        output = self.geneset_decoder(inv_latents)
        return {"recons": output, "latents": latents}


class PerturbationGeneSetAE(GeneSetAE):
    def __init(
        self,
        input_dim: int = 2613,
        latent_dim: int = 128,
        hidden_dims: List = [512, 512, 512, 512],
        batchnorm: bool = True,
        geneset_adjacencies: Tensor = None,
        geneset_adjacencies_file=None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            batchnorm=batchnorm,
            geneset_adjacencies=geneset_adjacencies,
            geneset_adjacencies_file=geneset_adjacencies_file,
        )
        self.eval()
        self.geneset_activities = None

    def encode(self, inputs: Tensor) -> Any:
        geneset_activities = self.geneset_encoder(inputs)
        latents = self.encoder(geneset_activities)
        return latents

    def decode(self, latents: Tensor) -> Any:
        decoded_geneset_activities = self.decoder(latents)
        recons = self.geneset_decoder(decoded_geneset_activities)
        return recons

    def forward(self, inputs: Tensor, silence_node: int) -> Any:
        geneset_activities = self.geneset_encoder(inputs)

        perturbed_geneset_activities = geneset_activities.clone()
        perturbed_geneset_activities[:, silence_node] = 0

        latents = self.encoder(geneset_activities)
        perturbed_latents = self.encoder(perturbed_geneset_activities)
        recons = self.decode(latents)
        perturbed_recons = self.decode(perturbed_latents)
        return {
            "recons": recons,
            "latents": latents,
            "geneset_activities": geneset_activities,
            "perturbed_latents": perturbed_latents,
            "perturbed_recons": perturbed_recons,
        }


class ImageToGeneSetTranslator(nn.Module):
    def __init__(self, image_ae: VanillaConvAE, geneset_ae: GeneSetAE):
        super().__init__()
        self.encoder = image_ae.encoder
        self.latent_mapper = image_ae.latent_mapper
        self.decoder = nn.Sequential(*list(geneset_ae.decoder.children())[:-1])
        self.decoder.add_module(
            "geneset_activation",
            nn.Sequential(
                *list(list(geneset_ae.decoder.children())[-1].children())[:-1]
            ),
        )

    def forward(self, inputs):
        encoded_inputs = self.encoder(inputs).view(inputs.size()[0], -1)
        latents = self.latent_mapper(encoded_inputs)
        geneset_activities = self.decoder(latents)
        return geneset_activities
