from typing import Tuple, List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from src.functions.loss_functions import max_discrepancy_loss
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
    device: str = "cuda:0",
) -> dict:
    if model_i.model_type != model_j.model_type:
        raise RuntimeError("Models must be of the same type, i.e. AE/AE or VAE/VAE.")
    if len(data_loader_i.dataset) != len(data_loader_j.dataset):
        raise RuntimeError(
            "Paired input data is required where sample k in dataset i refers to sample"
            " k in dataset j."
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
    knn_accs = {}
    for neighbors in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        knn_acc = 0.5 * (
            knn_accuracy(
                samples_i=latents_i, samples_j=latents_j, n_neighbours=neighbors
            )
            + knn_accuracy(
                samples_i=latents_j, samples_j=latents_i, n_neighbours=neighbors
            )
        )
        knn_accs[str(neighbors)] = knn_acc

    latent_l1_distance = max_discrepancy_loss(
        torch.from_numpy(latents_i).to(device), torch.from_numpy(latents_j).to(device)
    ).item()

    metrics = {"knn_accs": knn_accs, "latent_l1_distance": latent_l1_distance}
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
        device=device,
    )

    expanded_data = {}
    if "shared_latents" in data:
        shared_latents = data["shared_latents"]
        for i in range(shared_latents.shape[1]):
            expanded_data["zs_{}".format(i)] = shared_latents[:, i]
    if "domain_specific_latents" in data:
        domain_specific_latents = data["domain_specific_latents"]
        for i in range(domain_specific_latents.shape[1]):
            expanded_data["zd_{}".format(i)] = domain_specific_latents[:, i]
    if "labels" in data:
        expanded_data["labels"] = data["labels"]

    dict_to_csv(data=expanded_data, save_path=save_path)


def save_latents_to_csv(
    domain_config: DomainConfig,
    save_path: str,
    dataset_type: str = "val",
    device: str = "cuda:0",
):
    model = domain_config.domain_model_config.model
    try:
        dataset = domain_config.data_loader_dict[dataset_type].dataset
    except KeyError:
        raise RuntimeError(
            "Unknown dataset_type: {}, expected one of the following: train, val, test"
            .format(dataset_type)
        )
    save_latents_and_labels_to_csv(
        model=model,
        dataset=dataset,
        data_key=domain_config.data_key,
        label_key=domain_config.label_key,
        device=device,
        save_path=save_path,
    )


def get_full_latent_space_dict_for_multiple_domains(
    domain_configs: List[DomainConfig],
    dataset_type: str = "val",
    device: str = "cuda:0",
) -> Tuple[dict, dict]:
    latents_dict = {}
    label_dict = {}
    for domain_config in domain_configs:
        model = domain_config.domain_model_config.model
        try:
            dataset = domain_config.data_loader_dict[dataset_type].dataset
        except KeyError:
            raise RuntimeError(
                "Unknown dataset_type: {}, expected one of the following: train, val,"
                " test".format(dataset_type)
            )
        data = get_latent_representations_for_model(
            model=model,
            dataset=dataset,
            data_key=domain_config.data_key,
            label_key=domain_config.label_key,
            device=device,
        )

        shared_latents = data["shared_latents"]
        if "domain_specific_latents" in data:
            domain_specific_latents = data["domain_specific_latents"]
        else:
            domain_specific_latents = None

        latents_dict[domain_config.name] = shared_latents, domain_specific_latents

        if "labels" in data:
            label_dict[domain_config.name] = data["labels"]

    if len(label_dict) == 0:
        label_dict = None

    return latents_dict, label_dict


def get_shared_latent_space_dict_for_multiple_domains(
    domain_configs: List[DomainConfig],
    dataset_type: str = "val",
    device: str = "cuda:0",
) -> Tuple[dict, dict]:
    latents_dict = {}
    label_dict = {}
    for domain_config in domain_configs:
        model = domain_config.domain_model_config.model
        try:
            dataset = domain_config.data_loader_dict[dataset_type].dataset
        except KeyError:
            raise RuntimeError(
                "Unknown dataset_type: {}, expected one of the following: train, val,"
                " test".format(dataset_type)
            )
        data = get_latent_representations_for_model(
            model=model,
            dataset=dataset,
            data_key=domain_config.data_key,
            label_key=domain_config.label_key,
            device=device,
        )

        latents_dict[domain_config.name] = data["shared_latents"]
        if "labels" in data:
            label_dict[domain_config.name] = data["labels"]

    if len(label_dict) == 0:
        label_dict = None

    return latents_dict, label_dict


def evaluate_latent_clf(
    domain_configs: List[DomainConfig],
    latent_clf: torch.nn.Module,
    dataset_type: str = "test",
    device: str = "cuda:0",
):
    model_i = domain_configs[0].domain_model_config.model.to(device)
    model_j = domain_configs[1].domain_model_config.model.to(device)
    latent_clf.to(device)

    data_key_i = domain_configs[0].data_key
    label_key_i = domain_configs[0].label_key

    data_key_j = domain_configs[1].data_key
    label_key_j = domain_configs[1].label_key

    data_loader_i = domain_configs[0].data_loader_dict[dataset_type]
    data_loader_j = domain_configs[1].data_loader_dict[dataset_type]

    if model_i.model_type != model_j.model_type:
        raise RuntimeError("Models must be of the same type, i.e. AE/AE or VAE/VAE.")
    if len(data_loader_i.dataset) != len(data_loader_j.dataset):
        raise RuntimeError(
            "Paired input data is required where sample k in dataset i refers to sample"
            " k in dataset j."
        )

    # Store batch size to reset it after obtaining the latent representations.
    single_sample_loader_i = DataLoader(
        dataset=data_loader_i.dataset, shuffle=False, batch_size=1
    )
    single_sample_loader_j = DataLoader(
        dataset=data_loader_j.dataset, shuffle=False, batch_size=1
    )

    confusion_dict = {}
    labels_i = []
    labels_j = []
    preds_i = []
    preds_j = []

    for idx, (samples_i, samples_j) in enumerate(
        zip(single_sample_loader_i, single_sample_loader_j)
    ):
        input_i, input_j = samples_i[data_key_i], samples_j[data_key_j]
        label_i, label_j = samples_i[label_key_i], samples_j[label_key_j]

        input_i, input_j = input_i.to(device), input_j.to(device)
        latent_i = model_i(input_i)[1]
        latent_j = model_j(input_j)[1]

        _, pred_i = torch.max(latent_clf(latent_i), dim=1)
        _, pred_j = torch.max(latent_clf(latent_j), dim=1)

        labels_i.append(label_i.cpu().detach().numpy())
        labels_j.append(label_j.cpu().numpy())

        preds_i.append(pred_i.cpu().detach().numpy())
        preds_j.append(pred_j.cpu().detach().numpy())

    # Compute metrices
    labels_i = np.array(labels_i).squeeze()
    labels_j = np.array(labels_j).squeeze()

    preds_i = np.array(preds_i).squeeze()
    preds_j = np.array(preds_j).squeeze()

    confusion_dict[domain_configs[0].name] = confusion_matrix(labels_i, preds_i)
    confusion_dict[domain_configs[1].name] = confusion_matrix(labels_j, preds_j)
    confusion_dict["overall"] = confusion_matrix(
        np.concatenate([labels_i, labels_j]), np.concatenate([preds_i, preds_j])
    )
    return confusion_dict
