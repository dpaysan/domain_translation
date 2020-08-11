import logging
import os

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms import Compose

from src.utils.basic.io import get_data_list

class LabeledDataset(Dataset):
    def __init__(self):
        super(LabeledDataset, self).__init__()
        self.labels = None
        self.transformation_pipeline = None


class TorchNucleiImageDataset(LabeledDataset):
    def __init__(
            self, image_dir: str, label_fname: str, transform_pipeline: Compose = None,
    ):
        super(TorchNucleiImageDataset, self).__init__()

        self.image_dir = image_dir
        labels = pd.read_csv(label_fname, index_col=0)
        labels = labels.sort_values(by='nucleus_id')
        self.labels = np.array(labels.loc[:, 'binary_label'])
        self.image_locs = get_data_list(self.image_dir)
        self.transform_pipeline = transform_pipeline

    def __len__(self) -> int:
        return len(self.image_locs)

    def __getitem__(self, index: int) -> dict:
        img_loc = self.image_locs[index]
        image = self.process_image(image_loc=img_loc)
        if self.transform_pipeline is not None:
            image = self.transform_pipeline(image)

        label = np.array(self.labels[index]).astype(int)
        label = torch.from_numpy(label)

        sample = {"image": image, "label": label}
        return sample

    def set_transform_pipeline(
            self, transform_pipeline: transforms.Compose = None
    ) -> None:
        self.transform_pipeline = transform_pipeline

    def process_image(self, image_loc: str) -> Tensor:
        image = io.imread(image_loc)
        image = np.array(image, dtype=np.float32)
        image = torch.from_numpy(image).unsqueeze(0)
        return image


class TorchSeqDataset(LabeledDataset):
    def __init__(self, seq_data_and_labels_fname: str, transform_pipeline: Compose = None,
                 sample_index:bool=True):
        super(TorchSeqDataset, self).__init__()
        self.seq_data_and_labels_fname = seq_data_and_labels_fname
        seq_data_and_labels = pd.read_csv(self.seq_data_and_labels_fname, index_col=0)

        if sample_index:
            self.sample_ids = list(seq_data_and_labels.index)
        else:
            self.sample_ids = None

        self.seq_data = np.array(seq_data_and_labels.iloc[:, :-1]).astype(np.float32)
        self.labels = np.array(seq_data_and_labels.iloc[:, -1]).astype(int)
        self.transform_pipeline = transform_pipeline

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        seq_data = torch.from_numpy(self.seq_data[index])
        label = torch.from_numpy(np.array(self.labels[index]))
        sample = {'seq_data': seq_data, 'label': label}
        if self.sample_ids is not None:
            sample['id'] = self.sample_ids[index]
        return sample


class TorchTransformableSubset(Subset):
    def __init__(self, dataset: LabeledDataset, indices):
        super().__init__(dataset=dataset, indices=indices)

    def set_transform_pipeline(self, transform_pipeline: transforms.Compose) -> None:
        try:
            self.dataset.set_transform_pipeline(transform_pipeline)
        except AttributeError as exception:
            logging.error(
                "Object must implement a subset of a dataset type that implements the "
                "set_transform_pipeline method."
            )
            raise exception
