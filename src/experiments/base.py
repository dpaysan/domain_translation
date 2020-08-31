import os
from random import seed
from typing import List

import numpy as np
import torch

from src.utils.basic.visualization import (
    plot_train_val_hist,
    visualize_shared_latent_space,
    visualize_correlation_structure_latent_space,
)
from src.utils.torch.evaluation import save_latents_to_csv
from src.utils.torch.general import get_device, get_latent_distance_loss


class BaseExperiment:
    def __init__(
        self,
        output_dir: str,
        latent_clf_config: dict = None,
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
        self.latent_clf_config = latent_clf_config
        self.random_state = random_state
        self.loss_dict = None
        self.device = get_device()

        # Fix random seeds for reproducibility and limit applicable algorithms to those believed to be deterministic
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def visualize_loss_evolution(self):
        plot_train_val_hist(
            training_history=self.loss_dict["train"],
            validation_history=self.loss_dict["val"],
            output_dir=self.output_dir + "/",
            y_label="total_loss",
            title="Fitting history",
        )


class BaseTwoDomainExperiment(BaseExperiment):
    def __init__(
        self,
        output_dir: str,
        latent_clf_config: dict = None,
        latent_dcm_config: dict = None,
        train_val_test_split: List = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        num_epochs: int = 500,
        early_stopping: int = 20,
        random_state: int = 42,
        paired_data: bool = False,
        latent_distance_loss: str = None,
        latent_supervision_rate: float = 0.0,
    ):
        super().__init__(
            output_dir=output_dir,
            latent_clf_config=latent_clf_config,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            random_state=random_state,
        )

        self.latent_dcm_config = latent_dcm_config
        self.paired_data = paired_data
        if self.paired_data and latent_distance_loss is not None:
            self.latent_distance_loss = get_latent_distance_loss(latent_distance_loss)
        else:
            self.latent_distance_loss = None
        self.latent_supervision_rate = latent_supervision_rate

        self.domain_configs = None
        self.trained_models = None

    def visualize_shared_latent_space(
        self,
        reduction: str = "umap",
        dataset_type: str = "val",
        save_path: str = None,
        posfix: str = "",
    ):
        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                "shared_latent_space_" + dataset_type + posfix + ".png",
            )
        visualize_shared_latent_space(
            domain_configs=self.domain_configs,
            save_path=save_path,
            dataset_type=dataset_type,
            random_state=self.random_state,
            reduction=reduction,
            device=self.device,
        )

    def visualize_latent_space_correlation_structure(
        self, dataset_type: str = "val", save_path: str = None, posfix: str = ""
    ):
        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                "latent_space_correlation_" + dataset_type + posfix + ".png",
            )

        visualize_correlation_structure_latent_space(
            domain_configs=self.domain_configs,
            save_path=save_path,
            dataset_type=dataset_type,
            device=self.device,
        )

    def save_latents_to_csv(
        self,
        domain_id: int = 0,
        dataset_type: str = "val",
        save_path: str = None,
        posfix: str = "",
    ):
        domain_config = self.domain_configs[domain_id]
        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                "latents_" + domain_config.name + "_" + dataset_type + posfix + ".csv",
            )
        save_latents_to_csv(
            domain_config=domain_config,
            dataset_type=dataset_type,
            save_path=save_path,
            device=self.device,
        )


class BaseExperimentCV:
    def __init__(
        self,
        output_dir: str,
        n_folds: int = 4,
        latent_clf_config: dict = None,
        train_val_split: List = [0.8, 0.2],
        batch_size: int = 32,
        num_epochs: int = 500,
        early_stopping: int = 20,
        random_state: int = 42,
    ):
        self.output_dir = output_dir
        self.latent_clf_config = latent_clf_config
        self.n_folds = n_folds
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.device = get_device()

        self.loss_dicts = None
        self.trained_models = None

        # Fix random seeds for reproducibility and limit applicable algorithms to those believed to be deterministic
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def visualize_loss_evolution(self, fold_id, output_dir: str = None):
        if output_dir is None:
            output_dir = self.output_dir
        plot_train_val_hist(
            training_history=self.loss_dicts[fold_id]["train"],
            validation_history=self.loss_dicts[fold_id]["val"],
            output_dir=output_dir + "/",
            y_label="total_loss",
            title="Fitting history",
            posfix="_fold{}".format(fold_id + 1),
        )


class BaseTwoDomainExperimentCV(BaseExperimentCV):
    def __init__(
        self,
        output_dir,
        n_folds: int = 4,
        latent_clf_config: dict = None,
        latent_dcm_config: dict = None,
        train_val_split: List = [0.8, 0.2],
        batch_size: int = 32,
        num_epochs: int = 500,
        early_stopping: int = 20,
        random_state: int = 42,
        paired_data: bool = False,
        latent_distance_loss: str = None,
        latent_supervision_rate: float = 0.0,
    ):
        super().__init__(
            output_dir=output_dir,
            n_folds=n_folds,
            latent_clf_config=latent_clf_config,
            train_val_split=train_val_split,
            batch_size=batch_size,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            random_state=random_state,
        )

        self.latent_dcm_config = latent_dcm_config
        self.paired_data = paired_data
        if self.paired_data and latent_distance_loss is not None:
            self.latent_distance_loss = get_latent_distance_loss(latent_distance_loss)
        else:
            self.latent_distance_loss = None

        self.latent_supervision_rate = latent_supervision_rate

        self.domain_configs = None
        self.trained_models = None

    def visualize_shared_latent_space(
        self,
        fold_id: int,
        reduction: str = "umap",
        dataset_type: str = "val",
        save_path: str = None,
        posfix: str = "",
    ):

        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                "shared_latent_space_"
                + dataset_type
                + posfix
                + "_fold{}.png".format(fold_id),
            )
        visualize_shared_latent_space(
            domain_configs=self.domain_configs,
            save_path=save_path,
            dataset_type=dataset_type,
            random_state=self.random_state,
            reduction=reduction,
            device=self.device,
        )

    def save_latents_to_csv(
        self,
        fold_id: int,
        domain_id: int = 0,
        dataset_type: str = "val",
        save_path: str = None,
        posfix: str = "",
    ):
        domain_config = self.domain_configs[domain_id]
        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                "latents_"
                + domain_config.name
                + "_"
                + dataset_type
                + posfix
                + "_fold{}.csv".format(fold_id),
            )
        save_latents_to_csv(
            domain_config=domain_config,
            dataset_type=dataset_type,
            save_path=save_path,
            device=self.device,
        )
