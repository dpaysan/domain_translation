import logging
import os

from skimage import io
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms import Compose
import pandas as pd
import numpy as np
import torch
from torch import Tensor

from src.utils.basic.io import get_data_list


class TorchNucleiImageDataset(Dataset):
    def __init__(
            self, image_dir: str, label_fname: str, transform_pipeline: Compose = None,
    ):
        super(TorchNucleiImageDataset, self).__init__()

        self.image_dir = image_dir
        labels = pd.read_csv(label_fname, index_col=0)
        self.labels_dict = dict(
            zip(list(labels["nucleus_id"]), list(labels["binary_label"]))
        )
        self.image_locs = get_data_list(self.image_dir)

        if transform_pipeline is not None:
            self.transform_pipeline = transform_pipeline

    def __len__(self) -> int:
        return len(self.image_locs)

    def __getitem__(self, index: int) -> dict:
        img_loc = self.image_locs[index]
        nucleus_id = os.path.split(img_loc)[1][:, "."]

        image = self.process_image(image_loc=img_loc)
        if self.transform_pipeline is not None:
            image = self.transform_pipeline(image)

        label = self.labels_dict[nucleus_id]
        label = torch.LongTensor(label)

        sample = {"image": image, "label": label}
        return sample

    def set_transform_pipeline(
            self, transform_pipeline: transforms.Compose = None
    ) -> None:
        self.transform_pipeline = transform_pipeline

    def process_image(self, image_loc: str) -> Tensor:
        image = io.imread(image_loc)
        image = np.float32(image)
        image = torch.from_numpy(image)
        return image


class TorchSeqDataset(Dataset):
    def __init__(self, seq_data_and_labels_fname: str, transform_pipeline: Compose = None):
        super(TorchSeqDataset, self).__init__()
        self.seq_data_and_labels_fname = seq_data_and_labels_fname
        self.seq_data_and_labels = pd.read_csv(self.seq_data_and_labels_fname)

    def __len__(self) -> int:
        return len(self.seq_data_and_labels)

    def __getitem__(self, index: int) -> dict:
        seq_data = self.seq_data_and_labels.loc[index, self.seq_data_and_labels.columns != 'binary_label']
        seq_data = np.array(seq_data)
        seq_data = torch.from_numpy(seq_data)

        label = self.seq_data_and_labels.loc[index, 'binary_label']
        label = torch.LongTensor(label)

        sample = {'seq_data': seq_data, 'label': label}
        return sample


class TorchTransformableSubset(Subset):
    def __init__(self, dataset: Dataset, indices):
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
