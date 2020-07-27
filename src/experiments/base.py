from typing import List

from torch.optim import Adam
from torch.optim.rmsprop import RMSprop
from torch.optim.sgd import SGD

from src.utils.torch.data import (
    get_data_loader_dict,
    init_cell_dataset,
    stratified_train_val_test_split,
)
from src.utils.torch.exp import get_loss_dict
from src.utils.torch.general import get_transformation_dict_for_train_val_test


class BaseExperiment:
    def __init__(
        self,
        output_dir: str,
        data_dir: str,
        model_config: dict,
        loss_config: dict,
        optimizer_config: dict,
        train_val_test_split: List = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        n_epochs: int = 2000,
        early_stopping: int = 20,
        data_key: str = "image",
        label_key: str = "label",
        random_state: int = 42,
    ):
        # I/O attributes
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Training attributes
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.optimizer_config = optimizer_config
        self.train_val_test_split = train_val_test_split

        # Data attributes
        self.data_key = data_key
        self.label_key = label_key

        # Other attributes
        self.model_config = model_config
        self.loss_config = loss_config
        self.random_state = random_state

        self.model = None
        self.model_name = None

        self.optimizer = None
        self.loss_dict = None
        self.data_loader_dict = None

    def init_optimizer(self):
        if self.model is None:
            raise AttributeError("No initialized model found for attribute: model.")

        optimizer_type = self.optimizer_config.pop("type")
        self.optimizer_config["params"] = self.model.parameters()
        if optimizer_type == "sgd":
            self.optimizer = SGD(**self.optimizer_config)
        elif optimizer_type == "adamw":
            self.optimizer = Adam(**self.optimizer_config)
        elif optimizer_type == "rmsprop":
            self.optimizer = RMSprop(**self.optimizer_config)
        else:
            raise NotImplementedError(
                "Unknown optimizer type: {}".format(optimizer_type)
            )

    def init_loss(self):
        self.loss_dict = get_loss_dict(**self.loss_config)

    def init_data_loader_dict(self, label_weights: dict = None):
        transformation_dict = get_transformation_dict_for_train_val_test()
        cell_dataset = init_cell_dataset(
            data_dir=self.data_dir, label_weights=label_weights
        )
        train_dataset, val_dataset, test_dataset = stratified_train_val_test_split(
            dataset=cell_dataset,
            splits=self.train_val_test_split,
            random_state=self.random_state,
        )
        dataset_dict = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        self.data_loader_dict = get_data_loader_dict(
            dataset_dict=dataset_dict,
            batch_size=self.batch_size,
            transformation_dict=transformation_dict,
        )
