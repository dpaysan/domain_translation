from typing import Any, List
import torch

from torch import Tensor
from torch import nn


class GenesetEncoder(nn.Module):
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


class GenesetDecoder(nn.Module):
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


class GeneSetAE_v2(nn.Module):
    def __init__(self, adjacency_matrix: Tensor, hidden_dims: List = [1024, 512, 256]):
        super().__init__()
        self.adjacency_matrix = adjacency_matrix
        self.hidden_dims = hidden_dims

        self.geneset_encoder = GenesetEncoder(
            adjacency_matrix=adjacency_matrix, hidden_dims=hidden_dims
        )
        self.geneset_decoder = GenesetDecoder(
            adjacency_matrix=adjacency_matrix, hidden_dims=hidden_dims[::-1]
        )

    def forward(self, inputs: Tensor):
        geneset_activities = self.geneset_encoder(inputs)
        output = self.geneset_decoder(geneset_activities)
        return {'recons':output, 'latents':geneset_activities}
