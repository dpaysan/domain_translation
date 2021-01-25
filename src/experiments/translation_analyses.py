from typing import List

import numpy as np
import pandas as pd
import cv2
import torch

from src.experiments.image_seq_domain_translation import ImageSeqTranslationExperiment
from src.utils.torch.evaluation import analyze_geneset_perturbation_in_image
from src.utils.torch.visualization import visualize_geneset_perturbation_in_image


class ImageSeqTranslationExperimentAnalysis(ImageSeqTranslationExperiment):
    def __init__(
            self,
            output_dir: str,
            image_data_config: dict,
            image_model_config: dict,
            seq_data_config: dict,
            seq_model_config: dict,
            latent_dcm_config: dict = None,
            latent_structure_model_config: dict = None,
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
            output_dir,
            image_data_config,
            image_model_config,
            seq_data_config,
            seq_model_config,
            latent_dcm_config,
            latent_structure_model_config,
            num_epochs,
            early_stopping,
            train_val_test_split,
            batch_size,
            random_state,
            paired_data,
            latent_distance_loss,
            latent_supervision_rate,
        )

    def initialize_image_data_set(self):
        super().initialize_image_data_set()

    def initialize_seq_data_set(self):
        super().initialize_seq_data_set()

    def get_and_set_paired_training_idc(self):
        super().get_and_set_paired_training_idc()

    def initialize_image_data_loader_dict(self):
        super().initialize_image_data_loader_dict()

    def initialize_seq_data_loader_dict(self):
        super().initialize_seq_data_loader_dict()

    def initialize_image_domain_config(self, train_model: bool = True):
        super().initialize_image_domain_config()

    def initialize_seq_domain_config(self):
        super().initialize_seq_domain_config()

    def initialize_dcm_model(self):
        super().initialize_dcm_model()

    def initialize_clf_model(self):
        super().initialize_clf_model()

    def load_model_for_domain_config(self, weights_fname: str, id: int = 0):
        super().load_model_for_domain_config(weights_fname=weights_fname, id=id)

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
            latent_structure_model_weights_loc: str = None,
    ):
        super().load_pretrained_models(
            domain_model_1_weights_loc=domain_model_1_weights_loc,
            domain_model_2_weights_loc=domain_model_2_weights_loc,
            latent_dcm_weights_loc=latent_dcm_weights_loc,
            latent_structure_model_weights_loc=latent_structure_model_weights_loc,
        )

    def perform_artificial_knockout(self, sequence_domain_id: int=1, image_domain_id:int=0, silencing_node: int=0,
                                    dataloader_type:str='train'):
        sequence_domain = self.domain_configs[sequence_domain_id]
        image_domain = self.domain_configs[image_domain_id]
        seq_dataloader = sequence_domain.data_loader_dict[dataloader_type]
        sequence_ae = sequence_domain.domain_model_config.model
        image_ae = image_domain.domain_model_config.model
        seq_data_key = sequence_domain.data_key

        data_dict = analyze_geneset_perturbation_in_image(geneset_ae=sequence_ae, image_ae=image_ae,
                                                          seq_dataloader=seq_dataloader,
                                                          seq_data_key=seq_data_key, silencing_node=silencing_node)

        visualize_geneset_perturbation_in_image(data_dict=data_dict, output_dir=self.output_dir,
                                                silencing_node=silencing_node)


