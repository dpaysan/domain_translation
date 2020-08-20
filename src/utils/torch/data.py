import logging
from typing import List

from torchvision.transforms import Compose

from src.data.datasets import (
    TorchNucleiImageDataset,
    TorchSeqDataset,
)


def init_nuclei_image_dataset(
    image_dir: str,
    label_fname: str,
    transform_pipeline: Compose = None,
    paired_training_idc: List[int] = None,
) -> TorchNucleiImageDataset:
    logging.debug(
        "Load images set from {} and label information from {}.".format(
            image_dir, label_fname
        )
    )
    nuclei_dataset = TorchNucleiImageDataset(
        image_dir=image_dir,
        label_fname=label_fname,
        transform_pipeline=transform_pipeline,
        paired_training_idc=paired_training_idc,
    )
    logging.debug("Samples loaded: {}".format(len(nuclei_dataset)))
    return nuclei_dataset


def init_seq_dataset(
    seq_data_and_labels_fname: str,
    transform_pipeline: Compose = None,
    paired_training_idc: List[int] = None,
):
    logging.debug("Load sequence data set from {}.".format(seq_data_and_labels_fname))
    seq_dataset = TorchSeqDataset(
        seq_data_and_labels_fname=seq_data_and_labels_fname,
        transform_pipeline=transform_pipeline,
        paired_training_idc=paired_training_idc,
    )
    logging.debug("Samples loaded: {}".format(len(seq_dataset)))
    return seq_dataset
