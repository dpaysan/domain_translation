import os
from typing import List

from src.experiments.base import BaseExperiment
from src.models.vae import AugmentedVAE
from src.utils.basic.visualization import visualize_vae_performance
from src.utils.torch.exp import train_combined_architecture


class VaeExperiment(BaseExperiment):
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        model_config: dict,
        optimizer_config: dict,
        loss_config: dict,
        train_val_test_split: List = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        n_epochs: int = 2000,
        early_stopping: int = 20,
        data_key: str = "image",
        label_key: str = "label",
        random_state: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            output_dir=output_dir,
            model_config=model_config,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            data_key=data_key,
            label_key=label_key,
            random_state=random_state,
        )
        self.fitted_model = None
        self.loss_history = None

    def init_vae_model(self):
        model_type = self.model_config.pop("type")
        if model_type == "augmented_vae":
            self.model = AugmentedVAE(**self.model_config)
            self.model_name = "AugmentedVAE"
        else:
            raise NotImplementedError

    def init_optimizer(self):
        super().init_optimizer()

    def init_loss(self):
        super().init_loss()

    def init_data_loader_dict(self, label_weights: dict = None):
        super().init_data_loader_dict(label_weights=label_weights)

    def train_vae_model(self):
        self.model, self.loss_history = train_combined_architecture(
            model=self.model,
            data_loaders_dict=self.data_loader_dict,
            optimizer=self.optimizer,
            loss_dict=self.loss_dict,
            data_key=self.data_key,
            label_key=self.label_key,
            n_epochs=self.n_epochs,
            early_stopping=self.early_stopping,
            output_dir=self.output_dir,
        )

    def visualize_model_performance(
        self, label_dict: dict = None, postfix: str = "unfitted"
    ):
        input_fig, output_fig, samples_fig = visualize_vae_performance(
            vae_model=self.model,
            data_loader=self.data_loader_dict["test"],
            data_key=self.data_key,
            label_key=self.label_key,
            label_dict=label_dict,
        )

        os.makedirs(self.output_dir + "/visualization/", exist_ok=True)
        input_fig.savefig(
            self.output_dir + "/visualization/inputs_{}.png".format(postfix)
        )
        output_fig.savefig(
            self.output_dir + "/visualization/recons_{}.png".format(postfix)
        )
        if samples_fig is not None:
            samples_fig.savefig(
                self.output_dir + "/visualization/samples_{}.png".format(postfix)
            )
