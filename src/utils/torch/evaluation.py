import os
from typing import Tuple

from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

from src.helper.models import DomainConfig
from src.utils.basic.export import dict_to_csv
from src.utils.basic.metric import knn_accuracy


def evaluate_latent_integration(
    model_i: Module,
    model_j: Module,
    data_loader_i: DataLoader,
    data_loader_j: DataLoader,
    data_key_i: str = "seq_data",
    data_key_j: str = "seq_data",
    n_neighbours: int = 5,
    device: str = "cuda:0",
) -> dict:
    if model_i.model_type != model_j.model_type:
        raise RuntimeError("Models must be of the same type, i.e. AE/AE or VAE/VAE.")
    if len(data_loader_i.dataset) != len(data_loader_j.dataset):
        raise RuntimeError(
            "Paired input data is required where sample k in dataset i refers to sample k in dataset j."
        )

    # Store batch size to reset it after obtaining the latent representations.
    single_sample_loader_i = DataLoader(
        dataset=data_loader_i.dataset, shuffle=False, batch_size=1
    )
    single_sample_loader_j = DataLoader(
        dataset=data_loader_j.dataset, shuffle=False, batch_size=1
    )

    latents_i = []
    latents_j = []

    for idx, (samples_i, samples_j) in enumerate(
        zip(single_sample_loader_i, single_sample_loader_j)
    ):
        input_i, input_j = samples_i[data_key_i], samples_j[data_key_j]
        input_i, input_j = input_i.to(device), input_j.to(device)
        latent_i = model_i(input_i)[1].detach().cpu().numpy()
        latent_j = model_j(input_j)[1].detach().cpu().numpy()

        latents_i.append(latent_i)
        latents_j.append(latent_j)

    # Compute metrices
    latents_i = np.array(latents_i).squeeze()
    latents_j = np.array(latents_j).squeeze()

    knn_acc = knn_accuracy(
        samples_i=latents_i, samples_j=latents_j, n_neighbours=n_neighbours
    )
    latent_l1_distance = nn.L1Loss()(
        torch.from_numpy(latents_i), torch.from_numpy(latents_j)
    ).item()

    metrics = {"knn_acc": knn_acc, "latent_l1_distance": latent_l1_distance}
    return metrics


def get_latent_representations_for_model(
    model: Module,
    dataset: Dataset,
    data_key: str = "seq_data",
    label_key: str = "label",
    device: str = "cuda:0",
) -> dict:
    # create Dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    shared_latent_representations = []
    domain_specific_latent_representations = []
    labels = []
    model.eval().to(device)

    for (idx, sample) in enumerate(dataloader):
        input = sample[data_key].to(device)
        if label_key is not None:
            labels.append(sample[label_key].item())

        output = model(input)
        shared_latent_representation = output[1]
        shared_latent_representations.append(
            shared_latent_representation.detach().cpu().numpy()
        )
        if hasattr(model, "n_latent_spaces") and model.n_latent_spaces == 2:
            domain_specific_latent_representation = output[2]
            domain_specific_latent_representations.append(
                domain_specific_latent_representation.detach().cpu().numpy()
            )

    shared_latent_representations = np.array(shared_latent_representations).squeeze()
    domain_specific_latent_representations = np.array(
        domain_specific_latent_representations
    ).squeeze()
    labels = np.array(labels).squeeze()

    latent_dict = {"shared_latents": shared_latent_representations}

    if len(domain_specific_latent_representations) != 0:
        latent_dict["domain_specific_latents"] = domain_specific_latent_representations
    if len(labels) != 0:
        latent_dict["labels"] = labels

    return latent_dict


def save_latents_and_labels_to_csv(
    model: Module,
    dataset: Dataset,
    save_path: str,
    data_key: str = "seq_data",
    label_key: str = "label",
    device: str = "cuda:0",
):
    data = get_latent_representations_for_model(
            model=model,
            dataset=dataset,
            data_key=data_key,
            label_key=label_key,
            device=device,)

    expanded_data = {}
    if 'shared_latents' in data:
        shared_latents = data['shared_latents']
        for i in range(shared_latents.shape[1]):
            expanded_data['zs_{}'.format(i)] = shared_latents[:, i]
    if 'domain_specific_latents' in data:
        domain_specific_latents = data['domain_specific_latents']
        for i in range(domain_specific_latents.shape[1]):
            expanded_data['zd_{}'.format(i)] = domain_specific_latents[:,i]
    if 'labels' in data:
        expanded_data['labels'] = data['labels']

    dict_to_csv(data=expanded_data, save_path=save_path)


def save_latents_to_csv(domain_config:DomainConfig, save_path:str, dataset_type: str = "val",
    device:str='cuda:0'):
    model = domain_config.domain_model_config.model
    try:
        dataset = domain_config.data_loader_dict[dataset_type].dataset
    except KeyError:
        raise RuntimeError(
                "Unknown dataset_type: {}, expected one of the following: train, val, test".format(
                    dataset_type
                )
            )
    save_latents_and_labels_to_csv(
            model=model,
            dataset=dataset,
            data_key=domain_config.data_key,
            label_key=domain_config.label_key,
            device=device,
            save_path=save_path,
        )
