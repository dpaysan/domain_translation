from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
import numpy as np
import torch

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
