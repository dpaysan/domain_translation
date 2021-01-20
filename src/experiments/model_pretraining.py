import logging
from typing import List

import torch

from src.experiments.base import BaseExperiment
from src.helper.data import DataHandler
from src.utils.torch.data import (
    init_nuclei_image_dataset,
    init_seq_dataset,
)
from src.utils.torch.evaluation import (
    evaluate_latent_clf_two_domains,
    evaluate_latent_clf_one_domain,
)
from src.utils.torch.exp import train_val_test_loop_single_domain
from src.utils.torch.general import get_device
from src.utils.torch.model import (
    get_domain_configuration,
    get_latent_model_configuration,
)


class PretrainAeExperiment(BaseExperiment):
    def __init__(
        self,
        output_dir: str,
        data_config: dict,
        model_config: dict,
        domain_name: str,
        latent_structure_model_config: dict = None,
        train_val_test_split: List[float] = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        num_epochs: int = 64,
        early_stopping: int = -1,
        random_state: int = 42,
    ):
        super().__init__(
            output_dir=output_dir,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            random_state=random_state,
        )

        self.data_config = data_config
        self.model_config = model_config
        self.domain_name = domain_name
        self.latent_structure_model_config = latent_structure_model_config

        self.data_set = None
        self.data_transform_pipeline_dict = None
        self.data_loader_dict = None
        self.data_key = None
        self.label_key = None
        self.domain_config = None

        self.trained_models = None
        self.loss_dict = None

        self.device = get_device()

    def initialize_image_data_set(self):
        image_dir = self.data_config["image_dir"]
        label_fname = self.data_config["label_fname"]
        self.data_set = init_nuclei_image_dataset(
            image_dir=image_dir, label_fname=label_fname
        )
        self.data_key = self.data_config["data_key"]
        self.label_key = self.data_config["label_key"]

    def initialize_seq_data_set(self):
        seq_data_and_labels_fname = self.data_config["data_fname"]
        self.data_key = self.data_config["data_key"]
        self.label_key = self.data_config["label_key"]
        self.data_set = init_seq_dataset(
            seq_data_and_labels_fname=seq_data_and_labels_fname
        )

    def initialize_data_loader_dict(self, drop_last_batch: bool = False):
        dh = DataHandler(
            dataset=self.data_set,
            batch_size=self.batch_size,
            num_workers=0,
            random_state=self.random_state,
            transformation_dict=self.data_transform_pipeline_dict,
            drop_last_batch=drop_last_batch,
        )
        dh.stratified_train_val_test_split(splits=self.train_val_test_split)
        dh.get_data_loader_dict(shuffle=True)
        self.data_loader_dict = dh.data_loader_dict

    def initialize_domain_config(self):
        model_config = self.model_config["model_config"]
        optimizer_config = self.model_config["optimizer_config"]
        recon_loss_config = self.model_config["loss_config"]

        self.domain_config = get_domain_configuration(
            name=self.domain_name,
            model_dict=model_config,
            data_loader_dict=self.data_loader_dict,
            data_key=self.data_key,
            label_key=self.label_key,
            optimizer_dict=optimizer_config,
            recon_loss_fct_dict=recon_loss_config,
        )

    def initialize_clf_model(self):
        model_config = self.latent_structure_model_config["model_config"]
        optimizer_config = self.latent_structure_model_config["optimizer_config"]
        loss_config = self.latent_structure_model_config["loss_config"]
        self.latent_structure_model_config = get_latent_model_configuration(
            model_dict=model_config,
            optimizer_dict=optimizer_config,
            loss_dict=loss_config,
            device=self.device,
        )

    def train_models(
        self,
        alpha: float = 1.0,
        gamma: float = 1.0,
        lamb: float = 0.00000001,
        use_latent_structure_model: bool = False,
        save_freq: int = 50,
    ):
        self.trained_models, self.loss_dict = train_val_test_loop_single_domain(
            output_dir=self.output_dir,
            domain_config=self.domain_config,
            latent_structure_model_config=self.latent_structure_model_config,
            num_epochs=self.num_epochs,
            early_stopping=self.early_stopping,
            device=self.device,
            alpha=alpha,
            gamma=gamma,
            lamb=lamb,
            use_latent_structure_model=use_latent_structure_model,
            save_freq=save_freq,
        )

        if use_latent_structure_model:
            confusion_dict = evaluate_latent_clf_one_domain(
                domain_config=self.domain_config,
                latent_clf=self.latent_structure_model_config["model"],
                device=self.device,
                dataset_type="test",
            )
            logging.debug(
                "Confusion matrices of the latent classifier: %s".format(),
                confusion_dict,
            )

    def load_model(self, weights_fname):
        weights = torch.load(weights_fname)
        self.domain_config.domain_model_config.model.load_state_dict(weights)
