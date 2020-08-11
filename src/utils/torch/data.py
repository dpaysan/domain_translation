import logging
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose

from src.data.datasets import (
    TorchTransformableSubset,
    TorchNucleiImageDataset,
    TorchSeqDataset,
    LabeledDataset,
)


def init_nuclei_image_dataset(
    image_dir: str, label_fname: str, transform_pipeline: Compose = None,
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
    )
    logging.debug("Samples loaded: {}".format(len(nuclei_dataset)))
    return nuclei_dataset


def init_seq_dataset(
    seq_data_and_labels_fname: str, transform_pipeline: Compose = None
):
    logging.debug("Load sequence data set from {}.".format(seq_data_and_labels_fname))
    seq_dataset = TorchSeqDataset(
        seq_data_and_labels_fname=seq_data_and_labels_fname,
        transform_pipeline=transform_pipeline,
    )
    logging.debug("Samples loaded: {}".format(len(seq_dataset)))
    return seq_dataset


class DataHandler(object):
    def __init__(
        self,
        dataset: LabeledDataset,
        batch_size: int = 64,
        num_workers: int = 0,
        transformation_dict: dict = None,
        random_state: int = 42,
        drop_last_batch: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformation_dict = transformation_dict
        self.random_state = random_state
        self.drop_last_batch = drop_last_batch
        self.train_val_test_datasets_dict = None
        self.data_loader_dict = None

    def stratified_train_val_test_split(self, splits: Iterable) -> None:
        indices = np.array(list(range(len(self.dataset))))
        labels = np.array(self.dataset.labels)
        train_portion, val_portion, test_portion = splits[0], splits[1], splits[2]

        train_and_val_idc, test_idc = train_test_split(
            indices,
            test_size=test_portion,
            stratify=labels,
            random_state=self.random_state,
        )

        train_idc, val_idc = train_test_split(
            train_and_val_idc,
            test_size=val_portion / (1 - test_portion),
            stratify=labels[train_and_val_idc],
            random_state=self.random_state,
        )

        train_dataset = TorchTransformableSubset(
            dataset=self.dataset, indices=train_idc
        )
        val_dataset = TorchTransformableSubset(dataset=self.dataset, indices=val_idc)
        test_dataset = TorchTransformableSubset(dataset=self.dataset, indices=test_idc)

        self.train_val_test_datasets_dict = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }

    def get_data_loader_dict(self, shuffle: bool = True,) -> None:
        if self.transformation_dict is not None:
            for k, transform_pipeline in self.transformation_dict.items():
                self.train_val_test_datasets_dict[k].set_transform_pipeline(
                    transform_pipeline
                )
        data_loader_dict = {}
        for k, dataset in self.train_val_test_datasets_dict.items():
            data_loader_dict[k] = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=shuffle and k == "train",
                num_workers=self.num_workers,
                drop_last=self.drop_last_batch,
            )

        self.data_loader_dict = data_loader_dict
