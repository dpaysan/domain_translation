from abc import ABC
from typing import List

from torch import Tensor
from torch import nn


class LatentDiscriminator(nn.Module):
    def __init__(
        self, latent_dim: int = 128, hidden_dims:List=[1024, 1024, 1024], n_classes: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes

        model_modules = [
            nn.Sequential(nn.Linear(self.latent_dim + 1, hidden_dims[0]), nn.PReLU())
        ]
        for i in range(0, len(self.hidden_dims) - 1):
            model_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]), nn.PReLU()
                )
            )
        model_modules.append(nn.Linear(self.hidden_dims[-1], self.n_classes))
        self.model = nn.Sequential(*model_modules)

    def forward(self, input: Tensor) -> Tensor:
        output = self.model(input)
        return output


class LatentClassifier(nn.Module, ABC):
    def __init__(self, latent_dim: int = 128, n_classes: int = 2, hidden_dims:List=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        if hidden_dims is not None and len(hidden_dims) > 0:
            self.hidden_dims = hidden_dims
            model_components = [nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dims[0]))]
            for i in range(1, len(self.hidden_dims)):
                model_components.append(nn.Sequential(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]), nn.PReLU()))
            model_components.append(nn.Linear(self.hidden_dims[-1], self.n_classes))
            self.model = nn.Sequential(*model_components)
        else:
            self.model = nn.Linear(self.latent_dim, self.n_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)
