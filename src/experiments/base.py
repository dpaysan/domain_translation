from typing import List

from torch.optim import Adam
from torch.optim.rmsprop import RMSprop
from torch.optim.sgd import SGD
from src.utils.torch.general import get_transformation_dict_for_train_val_test


class BaseExperiment:
    def __init__(
        self,
        output_dir: str,
        train_val_test_split: List = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        num_epochs: int = 500,
        early_stopping: int = 20,
        random_state: int = 42,
    ):
        # I/O attributes
        self.output_dir = output_dir

        # Training attributes
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split

        # Other attributes
        self.random_state = random_state
