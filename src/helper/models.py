from typing import List, Callable

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch import Tensor


class DomainModelConfig(object):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        recon_loss_function: Module,
        inputs: Tensor = None,
        labels: Tensor = None,
        train: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.recon_loss_function = recon_loss_function
        self.inputs = inputs
        self.labels = labels
        self.train = train


class DomainConfig(object):
    def __init__(
        self,
        name: str,
        model: Module,
        optimizer: Optimizer,
        recon_loss_function: Module,
        data_loader_dict: dict,
        data_key: str,
        label_key: str,
    ):
        self.name = name
        self.domain_model_config = DomainModelConfig(
            model=model, optimizer=optimizer, recon_loss_function=recon_loss_function
        )
        self.data_loader_dict = data_loader_dict
        self.data_key = data_key
        self.label_key = label_key
