from typing import List, Callable

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch import Tensor


class DomainModelConfiguration(object):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        recon_loss_function: Module,
        inputs: Tensor,
        labels:Tensor=None,
        train: bool=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.recon_loss_function = recon_loss_function
        self.inputs=inputs
        self.labels = labels
        self.train = train