from abc import ABC
from typing import Any, List
import torch

from torch import Tensor
from torch import nn
import pandas as pd
import numpy as np

from src.models.ae import BaseAE


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
        self.adjacency_matrix = nn.Parameter(adjacency_matrix.transpose(0, 1), requires_grad=False)
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
    def __init__(self, input_dim:int, hidden_dims: List = [64, 32, 16], geneset_adjacencies: Tensor=None,
                 geneset_adjacencies_file:str=None, latent_dim:int=256, batchnorm:bool=True):
        super().__init__()
        self.input_dim = input_dim
        if geneset_adjacencies is not None:
            self.geneset_adjacencies = nn.Parameter(geneset_adjacencies, requires_grad=False)
        elif geneset_adjacencies_file is not None:
            geneset_adjacencies = pd.read_csv(geneset_adjacencies_file,
                                              index_col=0)
            self.geneset_adjacencies = nn.Parameter(torch.from_numpy(np.array(geneset_adjacencies)),
                                                    requires_grad=False)
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.geneset_encoder = GenesetEncoder(
            adjacency_matrix=self.geneset_adjacencies, hidden_dims=hidden_dims
        )
        self.geneset_decoder = GenesetDecoder(
            adjacency_matrix=self.geneset_adjacencies, hidden_dims=hidden_dims[::-1]
        )

        self.latent_mapper = nn.Linear(self.geneset_encoder.output_dim, self.latent_dim)
        if batchnorm:
            self.latent_mapper = nn.Sequential(self.latent_mapper, nn.BatchNorm1d(self.latent_dim))
        self.inv_latent_mapper = nn.Linear(self.latent_dim, self.geneset_decoder.input_dim)

    def forward(self, inputs: Tensor):
        geneset_activities = self.geneset_encoder(inputs)
        latents = self.latent_mapper(geneset_activities)
        inv_latents = self.inv_latent_mapper(latents)
        output = self.geneset_decoder(inv_latents)
        return {'recons':output, 'latents':latents}
