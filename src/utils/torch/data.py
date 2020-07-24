from typing import Iterable, Tuple, List
from src.data.datasets import TorchTransformableSubset, TorchNucleiImageDataset

import logging
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


def init_nuclei_image_dataset(
    image_dir:str,
    label_fname:str,
    transform_pipeline: Compose = None,
)->TorchNucleiImageDataset:
    logging.debug("Load images set from {} and label information from {}.".format(image_dir, label_fname))
    nuclei_dataset = TorchNucleiImageDataset(image_dir=image_dir, label_fname=label_fname, transform_pipeline=transform_pipeline)
    logging.debug("Samples loaded: {}".format(len(nuclei_dataset)))
    return nuclei_dataset


def stratified_train_val_test_split(
    dataset: Dataset, splits: Iterable, random_state: int
) -> Tuple[
    TorchTransformableSubset, TorchTransformableSubset, TorchTransformableSubset
]:
    indices = np.array(list(range(len(dataset))))
    labels = np.array(dataset.labels)

    train_portion, val_portion, test_portion = splits[0], splits[1], splits[2]

    train_and_val_idc, test_idc = train_test_split(
        indices, test_size=test_portion, stratify=labels, random_state=random_state
    )

    train_idc, val_idc = train_test_split(
        train_and_val_idc,
        test_size=val_portion / (1 - test_portion),
        stratify=labels[train_and_val_idc],
        random_state=random_state,
    )

    train_dataset = TorchTransformableSubset(dataset=dataset, indices=train_idc)
    val_dataset = TorchTransformableSubset(dataset=dataset, indices=val_idc)
    test_dataset = TorchTransformableSubset(dataset=dataset, indices=test_idc)

    return train_dataset, val_dataset, test_dataset


def get_data_loader_dict(
    dataset_dict: dict,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    transformation_dict: dict = None,
) -> dict:
    if transformation_dict is not None:
        for k, transform_pipeline in transformation_dict.items():
            dataset_dict[k].set_transform_pipeline(transform_pipeline)
    data_loader_dict = {}
    for k, dataset in dataset_dict.items():
        data_loader_dict[k] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return data_loader_dict
