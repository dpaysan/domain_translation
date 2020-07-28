from torch import Tensor
from torch import nn


class LatentDiscriminator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims=[1024, 1024, 1024, 1024],
        n_classes: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes

        model_modules = [nn.Linear(self.latent_dim, hidden_dims[0])]
        for i in range(0, len(self.hidden_dims) - 1):
            model_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]), nn.ReLU()
                )
            )
        model_modules.append(nn.Linear(self.hidden_dims[-1], self.n_classes))
        self.model = nn.Sequential(*model_modules)

    def forward(self, input: Tensor) -> Tensor:
        output = self.model(input)
        return output


class LinearClassifier(nn.Module):
    def __init__(self, latent_dim: int = 128, n_classes: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.model = nn.Linear(self.latent_dim, self.n_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)
