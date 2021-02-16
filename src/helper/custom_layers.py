from abc import ABC

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class SparseLinear(nn.Linear, ABC):
    def __init__(self, in_features: int, out_features: int, adj_matrix: Tensor):
        super().__init__(in_features=in_features, out_features=out_features)
        # Expects adjacency matrix to be of dimensions [in_features, out_features]
        if self.weight.size()[1] == adj_matrix.size()[0]:
            adj_matrix = torch.transpose(adj_matrix, 0, 1)
        self.adj_matrix = nn.Parameter(adj_matrix, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        output = F.linear(x, self.weight * self.adj_matrix, self.bias)
        return output
