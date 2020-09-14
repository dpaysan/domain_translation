from typing import Tuple

from torch import Tensor
from torch.nn import Module


class MixtureComponentInferenceNetwork(Module):
    def __init__(self, hidden_network: Module, output_layer: GumbleSoftmax):
        super().__init__()
        self.hidden_network = hidden_network
        self.output_layer = output_layer

    def forward(
        self, input: Tensor, temperature: float = 1.0, hard: float = 0.0
    ) -> Tuple[Tensor]:
        h = self.hidden_network(input)
        logits, prob, y = self.output_layer(h, temperature, hard)
        return logits, prob, y


class GaussianGenerativeNet(Module):
    def __init__(self,):
        pass
