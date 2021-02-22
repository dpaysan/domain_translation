from abc import ABC
from typing import List

from torch import Tensor
from torch import nn

from src.utils.torch.general import get_activation_module


class LatentDiscriminator(nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: List = [1024, 1024, 1024],
        n_classes: int = 2,
        trainable: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.trainable = trainable

        if hidden_dims is not None:
            model_modules = [
            nn.Sequential(
                nn.Linear(self.latent_dim + 20, hidden_dims[0]),
                nn.ReLU(),
                # nn.BatchNorm1d(self.hidden_dims[0])
            )
        ]
            for i in range(0, len(self.hidden_dims) - 1):
                model_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                    nn.ReLU(),
                    #nn.BatchNorm1d(self.hidden_dims[i+1])
                )
            )
            model_modules.append(nn.Linear(self.hidden_dims[-1], self.n_classes))
            self.model = nn.Sequential(*model_modules)
        else:
            self.model = nn.Linear(self.latent_dim + 20, self.n_classes)

    def forward(self, input: Tensor) -> Tensor:
        output = self.model(input)
        return output


class LatentClassifier(nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int = 128,
        n_classes: int = 2,
        hidden_dims: List = None,
        output_activation: str = None,
        trainable: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.output_activation = output_activation
        self.trainable = trainable

        if hidden_dims is not None and len(hidden_dims) > 0:
            self.hidden_dims = hidden_dims
            model_components = [
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dims[0]), nn.ReLU()
                )
            ]
            for i in range(1, len(self.hidden_dims)):
                model_components.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                        nn.PReLU(),
                    )
                )
            model_components.append(nn.Linear(self.hidden_dims[-1], self.n_classes))
            self.model = nn.Sequential(*model_components)
        else:
            self.model = nn.Linear(self.latent_dim, self.n_classes)

        if self.output_activation is not None:
            output_activation_module = get_activation_module(self.output_activation)
            self.model = nn.Sequential(self.model, output_activation_module)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class LatentRegressor(nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List = None,
        output_activation: str = None,
        trainable: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_activation = output_activation
        self.trainable = trainable

        if hidden_dims is not None and len(hidden_dims) > 0:
            self.hidden_dims = hidden_dims
            model_components = [
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dims[0]), nn.PReLU()
                )
            ]
            for i in range(1, len(self.hidden_dims)):
                model_components.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                        nn.PReLU(),
                    )
                )
            model_components.append(nn.Linear(self.hidden_dims[-1], self.n_classes))
            self.model = nn.Sequential(*model_components)
        else:
            self.model = nn.Linear(self.latent_dim, 1)

        if self.output_activation is not None:
            output_activation_module = get_activation_module(self.output_activation)
            self.model = nn.Sequential(self.model, output_activation_module)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)
