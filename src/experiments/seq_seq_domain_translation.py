import logging

from typing import List

import torch

from src.experiments.base import BaseTwoDomainExperiment
from src.utils.torch.data import (
    init_seq_dataset,
    DataHandler,
)


from src.utils.torch.exp import train_val_test_loop_two_domains
from src.utils.torch.model import (
    get_domain_configuration,
    get_latent_model_configuration,
)


class SeqSeqTranslationExperiment(BaseTwoDomainExperiment):
    def __init__(
        self,
        output_dir: str,
        seq_data_config_1: dict,
        seq_model_config_1: dict,
        seq_data_config_2: dict,
        seq_model_config_2: dict,
        latent_dcm_config: dict = None,
        latent_clf_config: dict = None,
        num_epochs: int = 500,
        early_stopping: int = 20,
        train_val_test_split: List[float] = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        random_state: int = 42,
        paired_data: bool = False,
    ):
        super().__init__(
            output_dir=output_dir,
            latent_clf_config=latent_clf_config,
            latent_dcm_config=latent_dcm_config,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            random_state=random_state,
            paired_data=paired_data,
        )

        self.seq_data_config_1 = seq_data_config_1
        self.seq_data_config_2 = seq_data_config_2
        self.seq_model_config_1 = seq_model_config_1
        self.seq_model_config_2 = seq_model_config_2

        self.seq_data_set_1 = None
        self.seq_data_transform_pipeline_dict_1 = None
        self.seq_data_loader_dict_1 = None
        self.seq_data_key_1 = None
        self.seq_label_key_1 = None

        self.seq_data_set_2 = None
        self.seq_data_transform_pipeline_dict_2 = None
        self.seq_data_loader_dict_2 = None
        self.seq_data_key_2 = None
        self.seq_label_key_2 = None

    def initialize_seq_data_set_1(self):
        seq_data_and_labels_fname = self.seq_data_config_1["data_fname"]
        self.seq_data_key_1 = self.seq_data_config_1["data_key"]
        self.seq_label_key_1 = self.seq_data_config_1["label_key"]
        self.seq_data_set_1 = init_seq_dataset(
            seq_data_and_labels_fname=seq_data_and_labels_fname
        )

    def initialize_seq_data_set_2(self):
        seq_data_and_labels_fname = self.seq_data_config_2["data_fname"]
        self.seq_data_key_2 = self.seq_data_config_2["data_key"]
        self.seq_label_key_2 = self.seq_data_config_2["label_key"]
        self.seq_data_set_2 = init_seq_dataset(
            seq_data_and_labels_fname=seq_data_and_labels_fname
        )

    def initialize_seq_data_loader_dict_1(self):
        dh = DataHandler(
            dataset=self.seq_data_set_1,
            batch_size=self.batch_size,
            num_workers=0,
            random_state=self.random_state,
            transformation_dict=self.seq_data_transform_pipeline_dict_1,
        )
        dh.stratified_train_val_test_split(splits=self.train_val_test_split)
        dh.get_data_loader_dict()
        self.seq_data_loader_dict_1 = dh.data_loader_dict

    def initialize_seq_data_loader_dict_2(self):
        dh = DataHandler(
            dataset=self.seq_data_set_2,
            batch_size=self.batch_size,
            num_workers=0,
            random_state=self.random_state,
            transformation_dict=self.seq_data_transform_pipeline_dict_2,
        )
        dh.stratified_train_val_test_split(splits=self.train_val_test_split)
        dh.get_data_loader_dict()
        self.seq_data_loader_dict_2 = dh.data_loader_dict

    def initialize_seq_domain_config_1(
        self, name: str = "RNA", train_model: bool = True
    ):
        if self.domain_configs is None:
            self.domain_configs = []

        model_config = self.seq_model_config_1["model_config"]
        optimizer_config = self.seq_model_config_1["optimizer_config"]
        recon_loss_config = self.seq_model_config_1["loss_config"]

        seq_domain_config = get_domain_configuration(
            name=name,
            model_dict=model_config,
            data_loader_dict=self.seq_data_loader_dict_1,
            data_key=self.seq_data_key_1,
            label_key=self.seq_label_key_1,
            optimizer_dict=optimizer_config,
            recon_loss_fct_dict=recon_loss_config,
            train_model=train_model,
        )
        self.domain_configs.append(seq_domain_config)

    def initialize_seq_domain_config_2(
        self, name: str = "ATAC", train_model: bool = True
    ):
        if self.domain_configs is None:
            self.domain_configs = []

        model_config = self.seq_model_config_2["model_config"]
        optimizer_config = self.seq_model_config_2["optimizer_config"]
        recon_loss_config = self.seq_model_config_2["loss_config"]

        seq_domain_config = get_domain_configuration(
            name=name,
            model_dict=model_config,
            data_loader_dict=self.seq_data_loader_dict_2,
            data_key=self.seq_data_key_2,
            label_key=self.seq_label_key_2,
            optimizer_dict=optimizer_config,
            recon_loss_fct_dict=recon_loss_config,
            train_model=train_model,
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
