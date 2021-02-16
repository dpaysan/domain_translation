import os
from typing import List

import imageio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.experiments.image_seq_domain_translation import ImageSeqTranslationExperiment
from src.models.custom_networks import ImageToGeneSetTranslator
from src.utils.torch.evaluation import (
    analyze_geneset_perturbation_in_image,
    analyze_guided_gradcam_for_genesets,
    get_geneset_activities_and_translated_images_sequences,
)
from src.utils.torch.visualization import (
    visualize_geneset_perturbation_in_image,
    visualize_geneset_guided_grad_cams,
)


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
        self.seq_domain_id = -1

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
        self.seq_domain_id = len(self.domain_configs) - 1

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

    def perform_artificial_knockout(
        self,
        sequence_domain_id: int = 1,
        image_domain_id: int = 0,
        silencing_node: int = 0,
        dataloader_type: str = "train",
    ):
        sequence_domain = self.domain_configs[sequence_domain_id]
        image_domain = self.domain_configs[image_domain_id]
        seq_dataloader = sequence_domain.data_loader_dict[dataloader_type]
        sequence_ae = sequence_domain.domain_model_config.model
        image_ae = image_domain.domain_model_config.model
        seq_data_key = sequence_domain.data_key

        data_dict = analyze_geneset_perturbation_in_image(
            geneset_ae=sequence_ae,
            image_ae=image_ae,
            seq_dataloader=seq_dataloader,
            seq_data_key=seq_data_key,
            silencing_node=silencing_node,
        )

        visualize_geneset_perturbation_in_image(
            data_dict=data_dict,
            output_dir=self.output_dir,
            silencing_node=silencing_node,
        )

    def analyze_pathway_activation_areas(
        self,
        target_layer: str = "3",
        sequence_domain_id: int = 1,
        image_domain_id: int = 0,
        query_node: int = 0,
        dataloader_type: str = "train",
    ):
        sequence_domain = self.domain_configs[sequence_domain_id]
        image_domain = self.domain_configs[image_domain_id]
        image_ae = image_domain.domain_model_config.model
        sequence_ae = sequence_domain.domain_model_config.model
        image_data_key = image_domain.data_key
        image_dataloader = image_domain.data_loader_dict[dataloader_type]

        image_geneset_translator = ImageToGeneSetTranslator(
            image_ae=image_ae, geneset_ae=sequence_ae
        )

        data_dict = analyze_guided_gradcam_for_genesets(
            image_to_geneset_translator=image_geneset_translator,
            image_dataloader=image_dataloader,
            image_data_key=image_data_key,
            query_node=query_node,
            target_layer=target_layer,
        )
        visualize_geneset_guided_grad_cams(
            data_dict=data_dict, output_dir=self.output_dir, query_node=query_node
        )

    def store_pathway_activities_and_translated_images_sequences(
        self, dataloader_type: str = "train", geneset_adjacencies_file: str = None
    ):
        data_dict = get_geneset_activities_and_translated_images_sequences(
            domain_configs=self.domain_configs, dataloader_type=dataloader_type
        )
        if geneset_adjacencies_file is not None:
            gene_pathway_adj = pd.read_csv(geneset_adjacencies_file, index_col=0)
            genes = list(gene_pathway_adj.index)
            pathways = list(gene_pathway_adj.columns)
        else:
            pathways = None
            genes = None
        rna_cell_ids = data_dict["rna_cell_ids"]
        rna_labels = data_dict["rna_labels"]
        rna_inputs = data_dict["rna_inputs"]
        rna_latents = data_dict["rna_latents"]
        pathway_activities = data_dict["geneset_activities"]
        translated_images = data_dict["translated_images"]
        translated_image_latents = data_dict["translated_image_latents"]

        image_cell_ids = data_dict["image_cell_ids"]
        image_labels = data_dict["image_labels"]
        image_inputs = data_dict["image_inputs"]
        image_latents = data_dict["image_latents"]
        translated_sequences = data_dict["translated_sequences"]
        translated_sequence_latents = data_dict["translated_sequence_latents"]
        translated_pathway_activities = data_dict["translated_geneset_activities"]

        pathway_activity_df = pd.DataFrame(
            np.array(pathway_activities), columns=pathways, index=rna_cell_ids
        )
        pathway_activity_df.to_csv(
            os.path.join(self.output_dir, "pathway_activities.csv")
        )

        rna_label_df = pd.DataFrame(
            np.array(rna_labels), columns=["label"], index=rna_cell_ids
        )
        rna_label_df.to_csv(os.path.join(self.output_dir, "rna_labels.csv"))

        rna_inputs_df = pd.DataFrame(
            np.array(rna_inputs), columns=genes, index=rna_cell_ids
        )
        rna_inputs_df.to_csv(os.path.join(self.output_dir, "rna_inputs.csv"))

        rna_latents_df = pd.DataFrame(np.array(rna_latents), index=rna_cell_ids)
        rna_latents_df.to_csv(os.path.join(self.output_dir, "rna_latents.csv"))

        image_label_df = pd.DataFrame(
            np.array(image_labels), columns=["label"], index=image_cell_ids
        )
        image_label_df.to_csv(os.path.join(self.output_dir, "image_labels.csv"))

        image_latents_df = pd.DataFrame(np.array(image_latents), index=image_cell_ids)
        image_latents_df.to_csv(os.path.join(self.output_dir, "image_latents.csv"))

        translated_pathway_activity_df = pd.DataFrame(
            np.array(translated_pathway_activities),
            columns=pathways,
            index=image_cell_ids,
        )
        translated_pathway_activity_df.to_csv(
            os.path.join(self.output_dir, "translated_pathway_activities.csv")
        )

        translated_sequences = pd.DataFrame(
            np.array(translated_sequences), index=image_cell_ids, columns=genes
        )
        translated_sequences.to_csv(
            os.path.join(self.output_dir, "translated_rna_sequences.csv")
        )

        translated_sequence_latents_df = pd.DataFrame(
            np.array(translated_sequence_latents), index=image_cell_ids
        )
        translated_sequence_latents_df.to_csv(
            os.path.join(self.output_dir, "translated_rna_latents.csv")
        )

        translated_image_latents_df = pd.DataFrame(
            np.array(translated_image_latents), index=rna_cell_ids
        )
        translated_image_latents_df.to_csv(
            os.path.join(self.output_dir, "translated_image_latents.csv")
        )

        translated_image_dir = os.path.join(self.output_dir, "translated_images")
        os.makedirs(translated_image_dir, exist_ok=True)
        for i in range(len(translated_images)):
            imageio.imwrite(
                os.path.join(translated_image_dir, "%s.jpg" % (rna_cell_ids[i])),
                translated_images[i].reshape(64, 64),
            )

        input_image_dir = os.path.join(self.output_dir, "input_images")
        os.makedirs(input_image_dir, exist_ok=True)
        for i in range(len(image_inputs)):
            imageio.imwrite(
                os.path.join(input_image_dir, "%s.jpg" % (image_cell_ids[i])),
                image_inputs[i].reshape(64, 64),
            )
