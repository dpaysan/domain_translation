import copy
from typing import Tuple, List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from src.data.datasets import TorchSeqDataset
from src.functions.loss_functions import max_discrepancy_loss
from src.helper.models import DomainConfig
from src.models.ae import GeneSetAE
from src.models.custom_networks import PerturbationGeneSetAE, ImageToGeneSetTranslator
from src.models.gradcam import GradCam, GuidedBackpropReLUModel
from src.utils.basic.export import dict_to_csv
from src.utils.basic.metric import knn_accuracy
from src.utils.torch.general import get_device
from umap import UMAP


def evaluate_latent_integration(
    model_i: Module,
    model_j: Module,
    data_loader_i: DataLoader,
    data_loader_j: DataLoader,
    data_key_i: str = "seq_data",
    data_key_j: str = "seq_data",
    device: str = "cuda:0",
) -> dict:
    if model_i.model_base_type != model_j.model_base_type:
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
        latent_i = model_i(input_i)["latents"].detach().cpu().numpy()
        latent_j = model_j(input_j)["latents"].detach().cpu().numpy()

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
    # latent_l1_distance = torch.nn.L1Loss()(torch.from_numpy(latents_i).to(device), torch.from_numpy(latents_j).to(
    # device))

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
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    shared_latent_representations = []
    domain_specific_latent_representations = []
    labels = []
    model.eval().to(device)

    for (idx, sample) in enumerate(dataloader):
        input = sample[data_key].to(device)
        if label_key is not None:
            labels.extend(sample[label_key].detach().cpu().numpy())

        output = model(input)
        shared_latent_representation = output["latents"]
        shared_latent_representations.extend(
            shared_latent_representation.detach().cpu().numpy()
        )
        # if hasattr(model, "n_latent_spaces") and model.n_latent_spaces == 2:
        if "unshared_latents" in output:
            domain_specific_latent_representation = output["unshared_latents"]
            domain_specific_latent_representations.extend(
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


def evaluate_latent_clf_one_domain(
    domain_config: DomainConfig,
    latent_clf: torch.nn.Module,
    dataset_type: str = "test",
    device: str = "cuda:0",
):
    model = domain_config.domain_model_config.model.to(device)
    latent_clf.to(device)

    data_key = domain_config.data_key
    label_key = domain_config.label_key

    data_loader = domain_config.data_loader_dict[dataset_type]

    single_sample_loader = DataLoader(
        dataset=data_loader.dataset, shuffle=False, batch_size=1
    )
    labels = []
    preds = []

    for idx, sample in enumerate(single_sample_loader):
        input = sample[data_key].to(device)
        label = sample[label_key].to(device)
        latent = model(input)["latents"]
        _, pred = torch.max(latent_clf(latent), dim=1)

        labels.append(label.cpu().numpy())
        preds.append(pred.cpu().detach().numpy())
    labels = np.array(labels).squeeze()
    preds = np.array(preds).squeeze()
    return confusion_matrix(labels, preds)


def evaluate_latent_clf_two_domains(
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

    if model_i.model_base_type != model_j.model_base_type:
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
        latent_i = model_i(input_i)["latents"]
        latent_j = model_j(input_j)["latents"]

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


def analyze_geneset_perturbation_in_image(
    geneset_ae: GeneSetAE,
    image_ae: Module,
    seq_dataloader: DataLoader,
    seq_data_key: str,
    silencing_node: int,
):
    device = get_device()
    geneset_ae.to(device).eval()
    image_ae.to(device).eval()
    perturbation_geneset_ae = PerturbationGeneSetAE(
        input_dim=geneset_ae.input_dim,
        latent_dim=geneset_ae.latent_dim,
        hidden_dims=geneset_ae.hidden_dims,
        geneset_adjacencies=geneset_ae.geneset_adjacencies,
    )
    perturbation_geneset_ae.to(device)
    perturbation_geneset_ae.load_state_dict(geneset_ae.state_dict())
    perturbation_geneset_ae.eval()
    geneset_ae.cpu()
    translated_images = []
    perturbed_translated_images = []
    recon_sequences = []
    perturbed_recon_sequences = []

    for (i, sample) in enumerate(seq_dataloader):
        inputs = sample[seq_data_key].to(device)
        output_dict = perturbation_geneset_ae(inputs, silencing_node)
        recon_sequences.extend(list(output_dict["recons"].detach().cpu().numpy()))
        latents = output_dict["latents"]
        geneset_activites = output_dict["geneset_activities"]
        perturbed_latents = output_dict["perturbed_latents"]
        perturbed_recon_sequences.extend(
            list(output_dict["perturbed_recons"].detach().cpu().numpy())
        )
        translated_images.extend(list(image_ae.decode(latents).detach().cpu().numpy()))
        perturbed_translated_images.extend(
            list(image_ae.decode(perturbed_latents).detach().cpu().numpy())
        )

    data_dict = {
        "seq_recons": np.array(recon_sequences),
        "perturbed_seq_recons": np.array(perturbed_recon_sequences),
        "trans_images": np.array(translated_images),
        "perturbed_trans_images": np.array(perturbed_translated_images),
    }
    return data_dict


def analyze_guided_gradcam_for_genesets(
    image_to_geneset_translator: ImageToGeneSetTranslator,
    image_dataloader: DataLoader,
    image_data_key: str,
    target_layer: str,
    query_node: int,
):
    device = get_device()
    image_to_geneset_translator.eval().to(device)

    grad_cam = GradCam(
        model=image_to_geneset_translator,
        feature_module=image_to_geneset_translator.encoder,
        target_layer_names=[target_layer],
        device=device,
    )
    gb_model = GuidedBackpropReLUModel(
        model=copy.deepcopy(image_to_geneset_translator), device=device
    )
    guided_backpropagation_maps = []
    gradient_cams = []
    images = []

    for (i, sample) in enumerate(image_dataloader):
        inputs = sample[image_data_key].to(device)
        gradient_cams.append(grad_cam(inputs, target_node=query_node).cpu().numpy())
        guided_backpropagation_maps.extend(
            list(gb_model(inputs, target_node=query_node))
        )
        images.extend(list(inputs.detach().cpu().numpy()))
    data_dict = {
        "images": np.array(images),
        "gb_maps": np.array(guided_backpropagation_maps),
        "grad_cams": np.array(gradient_cams),
    }
    return data_dict


def get_geneset_activities_and_translated_images_sequences(
    domain_configs: List[DomainConfig], dataloader_type: str
):
    if len(domain_configs) != 2:
        raise RuntimeError(
            "Expects two domain configurations (image and sequencing domain)"
        )
    if domain_configs[0].name == "image" and domain_configs[1].name == "rna":
        image_domain_config = domain_configs[0]
        rna_domain_config = domain_configs[1]
    elif domain_configs[0].name == "rna" and domain_configs[1].name == "image":
        image_domain_config = domain_configs[1]
        rna_domain_config = domain_configs[0]
    else:
        raise RuntimeError("Expected domain configuration types are >image< and >rna<.")

    rna_data_loader = rna_domain_config.data_loader_dict[dataloader_type]
    image_data_loader = image_domain_config.data_loader_dict[dataloader_type]
    device = get_device()

    rna_cell_ids = []
    all_rna_labels = []
    all_rna_inputs = []
    all_rna_latents = []
    all_geneset_activities = []
    all_reconstructed_geneset_activities = []
    all_reconstructed_rna_inputs = []
    all_translated_images = []
    all_image_latents = []

    image_cell_ids = []
    all_image_labels = []
    all_image_inputs = []
    all_translated_rna_seq = []
    all_translated_rna_latents = []
    all_translated_geneset_activities = []
    all_translated_image_latents = []

    geneset_ae = rna_domain_config.domain_model_config.model.to(device).eval()
    image_ae = image_domain_config.domain_model_config.model.to(device).eval()

    for i, sample in enumerate(rna_data_loader):
        rna_inputs = sample[rna_domain_config.data_key].to(device)
        rna_labels = sample[rna_domain_config.label_key]
        cell_ids = sample["id"]

        geneset_ae_output = geneset_ae(rna_inputs)
        latents = geneset_ae_output["latents"]
        geneset_activities = geneset_ae_output["geneset_activities"]
        reconstructed_geneset_activities = geneset_ae_output["decoded_geneset_activities"]
        reconstructed_rna_inputs = geneset_ae_output["recons"]
        translated_images = image_ae.decode(latents)
        translated_image_latents = image_ae(translated_images)["latents"]

        rna_cell_ids.extend(cell_ids)
        all_rna_labels.extend(list(rna_labels.clone().detach().cpu().numpy()))
        all_rna_inputs.extend(list(rna_inputs.clone().detach().cpu().numpy()))
        all_geneset_activities.extend(
            list(geneset_activities.clone().detach().cpu().numpy())
        )
        all_translated_images.extend(
            list(translated_images.clone().detach().cpu().numpy())
        )
        all_translated_image_latents.extend(
            list(translated_image_latents.clone().detach().cpu().numpy())
        )

        all_rna_latents.extend(list(latents.clone().detach().cpu().numpy()))

        all_reconstructed_geneset_activities.extend(list(reconstructed_geneset_activities.clone().detach().cpu().numpy()))
        all_reconstructed_rna_inputs.extend(list(reconstructed_rna_inputs.clone().detach().cpu().numpy()))

    for i, sample in enumerate(image_data_loader):
        image_inputs = sample[image_domain_config.data_key].to(device)
        image_labels = sample[image_domain_config.label_key].to(device)
        cell_ids = sample["id"]

        image_ae_output = image_ae(image_inputs)
        latents = image_ae_output["latents"]
        translated_sequences, translated_geneset_activities = geneset_ae.decode(latents)
        geneset_ae_output = geneset_ae(translated_sequences)
        translated_rna_latents = geneset_ae_output["latents"]

        image_cell_ids.extend(cell_ids)
        all_image_labels.extend(list(image_labels.clone().detach().cpu().numpy()))
        all_image_inputs.extend(list(image_inputs.clone().detach().cpu().numpy()))
        all_translated_geneset_activities.extend(
            list(translated_geneset_activities.clone().detach().cpu().numpy())
        )
        all_translated_rna_seq.extend(
            list(translated_sequences.clone().detach().cpu().numpy())
        )
        all_image_latents.extend(list(latents.clone().detach().cpu().numpy()))

        all_translated_rna_latents.extend(
            list(translated_rna_latents.clone().detach().cpu().numpy())
        )

    data_dict = {
        "rna_cell_ids": rna_cell_ids,
        "rna_labels": all_rna_labels,
        "rna_inputs": all_rna_inputs,
        "rna_latents": all_rna_latents,
        "geneset_activities": all_geneset_activities,
        "reconstructed_geneset_activities": all_reconstructed_geneset_activities,
        "reconstructed_rna_inputs":all_reconstructed_rna_inputs,
        "translated_images": all_translated_images,
        "translated_image_latents": all_translated_image_latents,
        "image_cell_ids": image_cell_ids,
        "image_labels": all_image_labels,
        "image_inputs": all_image_inputs,
        "image_latents": all_image_latents,
        "translated_sequences": all_translated_rna_seq,
        "translated_sequence_latents": all_translated_rna_latents,
        "translated_geneset_activities": all_translated_geneset_activities,
    }
    return data_dict


def perform_latent_walk_in_umap_space(
    domain_configs: List[DomainConfig], dataloader_type: str, random_state: int = 1234
):
    if len(domain_configs) != 2:
        raise RuntimeError(
            "Expects two domain configurations (image and sequencing domain)"
        )
    if domain_configs[0].name == "image" and domain_configs[1].name == "rna":
        image_domain_config = domain_configs[0]
        rna_domain_config = domain_configs[1]
    elif domain_configs[0].name == "rna" and domain_configs[1].name == "image":
        image_domain_config = domain_configs[1]
        rna_domain_config = domain_configs[0]
    else:
        raise RuntimeError("Expected domain configuration types are >image< and >rna<.")

    rna_data_loader = rna_domain_config.data_loader_dict[dataloader_type]
    image_data_loader = image_domain_config.data_loader_dict[dataloader_type]
    device = get_device()

    geneset_ae = rna_domain_config.domain_model_config.model.to(device).eval()
    image_ae = image_domain_config.domain_model_config.model.to(device).eval()

    all_rna_latents = []
    all_rna_labels = []
    all_image_latents = []
    all_image_labels = []
    grid_sequences = []
    grid_geneset_activities = []
    grid_images = []
    rna_cell_ids = []
    image_cell_ids = []

    for i, sample in enumerate(rna_data_loader):
        rna_inputs = sample[rna_domain_config.data_key].to(device)
        rna_labels = sample[rna_domain_config.label_key]
        rna_cell_ids.extend(sample["id"])

        geneset_ae_output = geneset_ae(rna_inputs)
        latents = geneset_ae_output["latents"]
        all_rna_latents.extend(list(latents.clone().detach().cpu().numpy()))
        all_rna_labels.extend(list(rna_labels.clone().detach().cpu().numpy()))

    for i, sample in enumerate(image_data_loader):
        image_inputs = sample[image_domain_config.data_key].to(device)
        image_labels = sample[image_domain_config.label_key].to(device)
        image_cell_ids.extend(sample["id"])

        image_ae_output = image_ae(image_inputs)
        latents = image_ae_output["latents"]
        all_image_latents.extend(list(latents.clone().detach().cpu().numpy()))
        all_image_labels.extend(list(image_labels.clone().detach().cpu().numpy()))

    all_latents = np.concatenate(
        (np.array(all_image_latents), np.array(all_rna_latents)), axis=0
    )
    all_labels = np.concatenate(
        (np.array(all_image_labels), np.array(all_rna_labels)), axis=0
    )
    all_domain_labels = np.concatenate(
        (
            np.repeat("image", len(all_image_labels)),
            np.repeat("rna", len(all_rna_labels)),
        ),
        axis=0,
    )
    all_cell_ids = np.concatenate((image_cell_ids, rna_cell_ids), axis=0)

    mapper = UMAP(random_state=random_state)
    transformed = mapper.fit_transform(all_latents)
    min_umap_c1 = min(transformed[:, 0])
    max_umap_c1 = max(transformed[:, 0])
    min_umap_c2 = min(transformed[:, 1])
    max_umap_c2 = max(transformed[:, 1])

    test_pts = np.array(
        [
            (np.array([min_umap_c1, max_umap_c2]) * (1 - x) + np.array([max_umap_c1, max_umap_c2]) * x)
            * (1 - y)
            + (np.array([min_umap_c1, min_umap_c2]) * (1 - x) + np.array([max_umap_c1, min_umap_c2]) * x)
            * y
            for y in np.linspace(0, 1, 10)
            for x in np.linspace(0, 1, 10)
        ]
    )

    inv_transformed_points = mapper.inverse_transform(test_pts)
    test_pts_ds = torch.utils.data.TensorDataset(torch.from_numpy(inv_transformed_points))
    test_pts_loader = torch.utils.data.DataLoader(
        test_pts_ds, batch_size=64, shuffle=False
    )

    for i, sample in enumerate(test_pts_loader):
        image_recons = image_ae.decode(sample[0].to(device))
        rna_recons, decoded_geneset_activities = geneset_ae.decode(sample[0].to(device))

        grid_images.extend(list(image_recons.clone().detach().cpu().numpy()))
        grid_sequences.extend(list(rna_recons.clone().detach().cpu().numpy()))
        grid_geneset_activities.extend(
            list(decoded_geneset_activities.clone().detach().cpu().numpy())
        )

    data_dict = {
        "grid_points": test_pts,
        "grid_images": grid_images,
        "grid_sequences": grid_sequences,
        "grid_geneset_activities": grid_geneset_activities,
        "all_latents": all_latents,
        "all_labels": all_labels,
        "all_domain_labels": all_domain_labels,
        "all_cell_ids": all_cell_ids,
    }

    return data_dict
