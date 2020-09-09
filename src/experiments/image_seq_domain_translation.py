import logging
from typing import List

import torch
import numpy as np

from src.experiments.base import BaseTwoDomainExperiment
from src.helper.data import DataHandler
from src.utils.torch.data import (
    init_nuclei_image_dataset,
    init_seq_dataset,
)
from src.utils.torch.exp import train_val_test_loop_two_domains
from src.utils.torch.model import (
    get_domain_configuration,
    get_latent_model_configuration,
)


class ImageSeqTranslationExperiment(BaseTwoDomainExperiment):
    def __init__(
        self,
        output_dir: str,
        image_data_config: dict,
        image_model_config: dict,
        seq_data_config: dict,
        seq_model_config: dict,
        latent_dcm_config: dict = None,
        latent_clf_config: dict = None,
        num_epochs: int = 500,
        early_stopping: int = 20,
        train_val_test_split: List[float] = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        random_state: int = 42,
        paired_data: bool = False,
        latent_distance_loss: str = None,
        latent_supervision_rate: float = 0.0,
    ):
        super().__init__(
            output_dir=output_dir,
            latent_dcm_config=latent_dcm_config,
            latent_clf_config=latent_clf_config,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            random_state=random_state,
            paired_data=paired_data,
            latent_distance_loss=latent_distance_loss,
            latent_supervision_rate=latent_supervision_rate,
        )

        self.image_data_config = image_data_config
        self.seq_data_config = seq_data_config

        self.image_data_set = None
        self.image_data_transform_pipeline_dict = None
        self.image_data_loader_dict = None
        self.image_data_key = None
        self.image_label_key = None

        self.seq_data_set = None
        self.seq_data_transform_pipeline_dict = None
        self.seq_data_loader_dict = None
        self.seq_data_key = None
        self.seq_label_key = None

        self.image_model_config = image_model_config
        self.seq_model_config = seq_model_config

    def initialize_image_data_set(self):
        image_dir = self.image_data_config["image_dir"]
        label_fname = self.image_data_config["label_fname"]
        self.image_data_set = init_nuclei_image_dataset(
            image_dir=image_dir, label_fname=label_fname
        )
        self.image_data_key = self.image_data_config["data_key"]
        self.image_label_key = self.image_data_config["label_key"]

    def initialize_seq_data_set(self):
        seq_data_and_labels_fname = self.seq_data_config["data_fname"]
        self.seq_data_key = self.seq_data_config["data_key"]
        self.seq_label_key = self.seq_data_config["label_key"]
        self.seq_data_set = init_seq_dataset(
            seq_data_and_labels_fname=seq_data_and_labels_fname
        )

    def get_and_set_paired_training_idc(self):
        np.random.seed(self.random_state)
        if self.seq_data_set is None or self.image_data_set is None:
            raise RuntimeError("Datasets must be initialized!")
        if len(self.seq_data_set) != len(self.image_data_set):
            raise RuntimeError(
                "Supervised training in the latent space requires paired data matching"
                " in the samples-dimension"
            )
        else:
            n_samples = len(self.seq_data_set)
            paired_training_idc = np.random.choice(
                list(range(n_samples)),
                size=int(n_samples * self.latent_supervision_rate),
                replace=False,
            )
            self.seq_data_set.paired_training_idc = paired_training_idc
            self.image_data_set.paired_training_idc = paired_training_idc

    def initialize_image_data_loader_dict(self):
        dh = DataHandler(
            dataset=self.image_data_set,
            batch_size=self.batch_size,
            num_workers=0,
            random_state=self.random_state,
            transformation_dict=self.image_data_transform_pipeline_dict,
        )
        dh.stratified_train_val_test_split(splits=self.train_val_test_split)
        dh.get_data_loader_dict(
            shuffle=self.latent_distance_loss is None
            or self.latent_supervision_rate == 0
        )
        self.image_data_loader_dict = dh.data_loader_dict

    def initialize_seq_data_loader_dict(self):
        dh = DataHandler(
            dataset=self.seq_data_set,
            batch_size=self.batch_size,
            num_workers=0,
            random_state=self.random_state,
            transformation_dict=self.seq_data_transform_pipeline_dict,
        )
        dh.stratified_train_val_test_split(splits=self.train_val_test_split)
        dh.get_data_loader_dict(
            shuffle=self.latent_distance_loss is None
            or self.latent_supervision_rate == 0
        )
        self.seq_data_loader_dict = dh.data_loader_dict

    def initialize_image_domain_config(self, train_model: bool = True):
        if self.domain_configs is None:
            self.domain_configs = []

        model_config = self.image_model_config["model_config"]
        optimizer_config = self.image_model_config["optimizer_config"]
        recon_loss_config = self.image_model_config["loss_config"]

        image_domain_config = get_domain_configuration(
            name="image",
            model_dict=model_config,
            data_loader_dict=self.image_data_loader_dict,
            data_key=self.image_data_key,
            label_key=self.image_label_key,
            optimizer_dict=optimizer_config,
            recon_loss_fct_dict=recon_loss_config,
            train_model=train_model,
        )
        self.domain_configs.append(image_domain_config)

    def initialize_seq_domain_config(self):
        if self.domain_configs is None:
            self.domain_configs = []

        model_config = self.seq_model_config["model_config"]
        optimizer_config = self.seq_model_config["optimizer_config"]
        recon_loss_config = self.seq_model_config["loss_config"]

        seq_domain_config = get_domain_configuration(
            name="rna",
            model_dict=model_config,
            data_loader_dict=self.seq_data_loader_dict,
            data_key=self.seq_data_key,
            label_key=self.seq_label_key,
            optimizer_dict=optimizer_config,
            recon_loss_fct_dict=recon_loss_config,
        )
        self.domain_configs.append(seq_domain_config)

    def initialize_dcm_model(self):
        model_config = self.latent_dcm_config["model_config"]
        optimizer_config = self.latent_dcm_config["optimizer_config"]
        loss_config = self.latent_dcm_config["loss_config"]
        self.latent_dcm_config = get_latent_model_configuration(
            model_dict=model_config,
            optimizer_dict=optimizer_config,
            loss_dict=loss_config,
            device=self.device,
        )

    def initialize_clf_model(self):
        model_config = self.latent_clf_config["model_config"]
        optimizer_config = self.latent_clf_config["optimizer_config"]
        loss_config = self.latent_clf_config["loss_config"]
        self.latent_clf_config = get_latent_model_configuration(
            model_dict=model_config,
            optimizer_dict=optimizer_config,
            loss_dict=loss_config,
            device=self.device,
        )

    def load_model_for_domain_config(self, weights_fname: str, id: int = 0):
        model_state_dict = torch.load(weights_fname)
        self.domain_configs[id].domain_model_config.model.load_state_dict(
            model_state_dict
        )
        logging.debug(
            "Model loaded from {} and set for domain {}.".format(
                weights_fname, self.domain_configs[id].name
            )
        )

    def train_models(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        lamb: float = 0.00000001,
        use_dcm: bool = True,
        use_clf: bool = False,
        save_freq: int = 50,
    ):
        self.trained_models, self.loss_dict = train_val_test_loop_two_domains(
            output_dir=self.output_dir,
            domain_configs=self.domain_configs,
            latent_dcm_config=self.latent_dcm_config,
            latent_clf_config=self.latent_clf_config,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            lamb=lamb,
            use_dcm=use_dcm,
            use_clf=use_clf,
            num_epochs=self.num_epochs,
            save_freq=save_freq,
            early_stopping=self.early_stopping,
            device=self.device,
            paired_mode=self.paired_data,
        )

    def save_latents_to_csv(
        self,
        domain_id: int = 0,
        dataset_type: str = "val",
        save_path: str = None,
        posfix: str = "",
    ):
        super().save_latents_to_csv(
            domain_id=domain_id,
            dataset_type=dataset_type,
            save_path=save_path,
            posfix=posfix,
        )

    def visualize_loss_evolution(self):
        super().visualize_loss_evolution()

    def visualize_shared_latent_space(
        self,
        reduction: str = "umap",
        dataset_type: str = "val",
        save_path: str = None,
        posfix: str = "",
    ):
        super().visualize_shared_latent_space(
            reduction=reduction,
            dataset_type=dataset_type,
            save_path=save_path,
            posfix=posfix,
        )

    def load_pretrained_models(
        self,
        domain_model_1_weights_loc: str,
        domain_model_2_weights_loc: str,
        latent_dcm_weights_loc: str,
        latent_clf_weights_loc: str = None,
    ):

        super().load_pretrained_models(
            domain_model_1_weights_loc=domain_model_1_weights_loc,
            domain_model_2_weights_loc=domain_model_2_weights_loc,
            latent_dcm_weights_loc=latent_dcm_weights_loc,
            latent_clf_weights_loc=latent_clf_weights_loc,
        )
