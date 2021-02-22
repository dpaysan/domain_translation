import copy
import logging
import os
import time
from itertools import cycle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from src.functions.metric import accuracy
from src.helper.models import DomainModelConfig, DomainConfig
from src.utils.basic.visualization import visualize_model_performance
from src.utils.torch.evaluation import evaluate_latent_integration
from src.utils.torch.general import get_device
from src.utils.torch.visualization import (
    visualize_image_translation_performance,
    visualize_image_vae_performance,
    visualize_image_ae_performance,
)


def train_autoencoders_two_domains(
    domain_model_configurations: List[DomainModelConfig],
    latent_dcm: Module,
    latent_dcm_loss: Module,
    latent_structure_model: Module = None,
    latent_structure_model_optimizer: Optimizer = None,
    latent_structure_model_loss: Module = None,
    alpha: float = 0.1,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    lamb: float = 0.000001,
    device: str = "cuda:0",
    use_latent_discriminator: bool = True,
    use_latent_structure_model: bool = True,
    phase: str = "train",
    model_base_type: str = "ae",
    latent_distance_loss: Module = None,
    paired_training_mask: Tensor = None,
) -> dict:
    # Expects 2 model configurations (one for each domain)
    model_configuration_i = domain_model_configurations[0]
    model_configuration_j = domain_model_configurations[1]

    # Get all parameters of the configuration for domain i
    model_i = model_configuration_i.model
    optimizer_i = model_configuration_i.optimizer
    inputs_i = model_configuration_i.inputs
    labels_i = model_configuration_i.labels
    recon_loss_fct_i = model_configuration_i.recon_loss_function
    train_i = model_configuration_i.trainable

    # Get all parameters of the configuration for domain j
    model_j = model_configuration_j.model
    optimizer_j = model_configuration_j.optimizer
    inputs_j = model_configuration_j.inputs
    labels_j = model_configuration_j.labels
    recon_loss_fct_j = model_configuration_j.recon_loss_function
    train_j = model_configuration_j.trainable

    # Set VAE models to train if defined in respective configuration
    if phase == "train" and train_i:
        model_i.train()
    else:
        model_i.eval()

    model_i.to(device)
    optimizer_i.zero_grad()

    if phase == "train" and train_j:
        model_j.train()
    else:
        model_j.eval()

    model_j.to(device)
    optimizer_j.zero_grad()

    # The discriminator will not be trained but only used to compute the adversarial loss for the AE updates
    latent_dcm.eval()
    latent_dcm.to(device)
    latent_dcm.zero_grad()

    if use_latent_structure_model:
        assert latent_structure_model is not None
        if phase == "train" and latent_structure_model.trainable:
            latent_structure_model.train()
        else:
            latent_structure_model.eval()
        latent_structure_model.to(device)
        latent_structure_model_optimizer.zero_grad()

    # Forward pass of the AE/VAE
    inputs_i, inputs_j = Variable(inputs_i).to(device), Variable(inputs_j).to(device)
    if labels_i is not None and labels_j is not None:
        labels_i, labels_j = (
            Variable(labels_i).to(device),
            Variable(labels_j).to(device),
        )

    outputs_i = model_i(inputs_i)
    outputs_j = model_j(inputs_j)

    recons_i = outputs_i["recons"]
    recons_j = outputs_j["recons"]

    latents_i = outputs_i["latents"]
    latents_j = outputs_j["latents"]

    if model_base_type == "vae":
        mu_i = outputs_i["mu"]
        mu_j = outputs_j["mu"]

        logvar_i = outputs_i["logvar"]
        logvar_j = outputs_j["logvar"]

        loss_dict_i = model_i.loss_function(
            inputs=inputs_i, recons=recons_i, mu=mu_i, logvar=logvar_i
        )
        loss_dict_j = model_j.loss_function(
            inputs=inputs_j, recons=recons_j, mu=mu_j, logvar=logvar_j
        )

        recon_loss_i = loss_dict_i["recon_loss"]
        kld_loss_i = loss_dict_i["kld_loss"]

        recon_loss_j = loss_dict_j["recon_loss"]
        kld_loss_j = loss_dict_j["kld_loss"]

        kl_loss = kld_loss_i + kld_loss_j
        total_loss = alpha * (recon_loss_i + recon_loss_j) + kl_loss * lamb
    elif model_base_type == "gmvae":

        logits_i = outputs_i["logits"]
        probs_i = outputs_i["probs"]
        component_labels_i = outputs_i["component_labels"]
        mu_i = outputs_i["mu"]
        logvar_i = outputs_i["logvar"]
        mu_component_prior_i = outputs_i["mu_component_prior"]
        logvar_component_prior_i = outputs_i["logvar_component_prior"]

        logits_j = outputs_j["logits"]
        probs_j = outputs_j["probs"]
        component_labels_j = outputs_j["component_labels"]
        mu_j = outputs_j["mu"]
        logvar_j = outputs_j["logvar"]
        mu_component_prior_j = outputs_j["mu_component_prior"]
        logvar_component_prior_j = outputs_j["logvar_component_prior"]

        loss_dict_i = model_i.loss_function(
            inputs=inputs_i,
            recons=recons_i,
            mu=mu_i,
            logvar=logvar_i,
            mu_prior=mu_component_prior_i,
            logvar_prior=logvar_component_prior_i,
            y_probs=probs_i,
            y_logits=logits_i,
            y_true=labels_i,
        )

        loss_dict_j = model_j.loss_function(
            inputs=inputs_j,
            recons=recons_j,
            mu=mu_j,
            logvar=logvar_j,
            mu_prior=mu_component_prior_j,
            logvar_prior=logvar_component_prior_j,
            y_probs=probs_j,
            y_logits=logits_j,
            y_true=labels_j,
        )

        recon_loss_i = loss_dict_i["recon_loss"]
        kld_loss_i = loss_dict_i["kld_loss"]
        component_prior_loss_i = loss_dict_i["component_prior_loss"]
        component_supervision_loss_i = loss_dict_i["component_supervision_loss"]

        recon_loss_j = loss_dict_j["recon_loss"]
        kld_loss_j = loss_dict_j["kld_loss"]
        component_prior_loss_j = loss_dict_j["component_prior_loss"]
        component_supervision_loss_j = loss_dict_j["component_supervision_loss"]

        kl_loss = kld_loss_i + kld_loss_j

        total_loss = alpha * (recon_loss_i + recon_loss_j) + lamb * kl_loss
        if component_prior_loss_i is not None and component_prior_loss_j is not None:
            total_loss += component_prior_loss_i + component_prior_loss_j
        elif (
            component_supervision_loss_i is not None
            and component_supervision_loss_j is not None
        ):
            total_loss += component_supervision_loss_i + component_supervision_loss_j

    elif model_base_type == "ae":
        recon_loss_i = model_i.loss_function(inputs=inputs_i, recons=recons_i)[
            "recon_loss"
        ]
        recon_loss_j = model_j.loss_function(inputs=inputs_j, recons=recons_j)[
            "recon_loss"
        ]
        total_loss = alpha * (recon_loss_i + recon_loss_j)
    else:
        raise RuntimeError("Unknown model type: {}".format(model_base_type))

    if use_latent_discriminator:

        # Add class label to latent representations to ensure that latent representations encode generic information
        # independent from the used group of the samples (see Adversarial AutoEncoder paper)
        dcm_input_i = torch.cat(
            (latents_i, labels_i.float().view(-1, 1).expand(-1, 20)), dim=1
        )
        dcm_input_j = torch.cat(
            (latents_j, labels_j.float().view(-1, 1).expand(-1, 20)), dim=1
        )

    else:
        dcm_input_i = latents_i
        dcm_input_j = latents_j

    dcm_output_i = latent_dcm(dcm_input_i)
    dcm_output_j = latent_dcm(dcm_input_j)

    domain_labels_i = torch.zeros(dcm_output_i.size(0)).long().to(device)
    domain_labels_j = torch.ones(dcm_output_j.size(0)).long().to(device)
    # domain_labels_i = (
    #     torch.ones(dcm_output_i.size(0)).float().to(device).view(-1, 1) * 0.1
    # )
    # domain_labels_j = (
    #     torch.ones(dcm_output_j.size(0)).float().to(device).view(-1, 1) * 0.9
    # )

    # Forward pass latent structure model if it is supposed to be trained and used to assess the integration of the
    # learned
    # latent spaces
    if use_latent_structure_model:
        latent_structure_model_output_i = latent_structure_model(latents_i)
        latent_structure_model_output_j = latent_structure_model(latents_j)

    # Calculate adversarial loss - by mixing labels indicating domain with output predictions to "confuse" the
    # discriminator and encourage learning autoencoder that make the distinction between the modalities in the latent
    # space as difficult as possible for the discriminator

    dcm_loss = 0.5 * latent_dcm_loss(
        dcm_output_i, domain_labels_j
    ) + 0.5 * latent_dcm_loss(dcm_output_j, domain_labels_i)

    total_loss += dcm_loss * beta

    # Add loss of latent structure model if this is trained
    if use_latent_structure_model:
        latent_sm_loss = 0.5 * (
            latent_structure_model_loss(latent_structure_model_output_i, labels_i)
            + latent_structure_model_loss(latent_structure_model_output_j, labels_j)
        )
        total_loss += latent_sm_loss * gamma
        latent_sm_acc_i = accuracy(latent_structure_model_output_i, labels_i)
        latent_sm_acc_j = accuracy(latent_structure_model_output_j, labels_j)

    # Add loss measuring the distance between a pair of samples in the latent space if this is desired
    # Be careful using this option as it is important that the samples in the batch are actually paired
    if paired_training_mask is not None:
        paired_distance_samples = paired_training_mask.sum().item()
        if paired_distance_samples > 0:
            paired_supervision_loss = latent_distance_loss(
                latents_i[paired_training_mask], latents_j[paired_training_mask]
            )
            total_loss += paired_supervision_loss * delta
        else:
            paired_supervision_loss = 0

    # Backpropagate loss and update parameters if we are in the training phase
    if phase == "train":
        total_loss.backward()
        if train_i:
            optimizer_i.step()
            model_i.updated = True
        if train_j:
            optimizer_j.step()
            model_j.updated = True
        if use_latent_structure_model:
            latent_structure_model_optimizer.step()

    # Get summary statistics
    batch_size_i = inputs_i.size(0)
    batch_size_j = inputs_j.size(0)
    summary_stats = {
        "recon_loss_i": recon_loss_i.item() * batch_size_i,
        "recon_loss_j": recon_loss_j.item() * batch_size_j,
        "dcm_loss": dcm_loss.item() * (batch_size_i + batch_size_j),
    }
    total_loss_item = (
        alpha * (summary_stats["recon_loss_i"] + summary_stats["recon_loss_j"])
        + summary_stats["dcm_loss"] * beta
    )
    if model_base_type == "vae":
        summary_stats["kl_loss"] = kl_loss.item()
        total_loss_item += kl_loss.item() * lamb
    elif model_base_type == "gmvae":
        summary_stats["kl_loss"] = kl_loss.item()
        total_loss_item = total_loss.item()

    if use_latent_structure_model:
        summary_stats["latent_structure_model_loss"] = latent_sm_loss.item() * (
            batch_size_i + batch_size_j
        )
        total_loss_item += summary_stats["latent_structure_model_loss"] * gamma
        summary_stats["latent_structure_model_accuracy_i"] = latent_sm_acc_i
        summary_stats["latent_structure_model_accuracy_j"] = latent_sm_acc_j

    if paired_training_mask is not None and paired_distance_samples > 0:
        summary_stats["latent_distance_loss"] = (
            paired_supervision_loss.item() * paired_distance_samples
        )
        summary_stats["paired_distance_samples"] = paired_distance_samples

    summary_stats["total_loss"] = total_loss_item

    return summary_stats


def train_latent_dcm_two_domains(
    domain_model_configurations: List[DomainModelConfig],
    latent_dcm: nn.Module,
    latent_dcm_optimizer: Optimizer,
    latent_dcm_loss: Module,
    use_latent_discriminator: bool,
    device: str = "cuda:0",
    phase: str = "train",
) -> dict:
    # Get the model configurations for the two domains
    model_configuration_i = domain_model_configurations[0]
    model_configuration_j = domain_model_configurations[1]

    # Get all parameters of the configuration for domain i
    model_i = model_configuration_i.model
    inputs_i = model_configuration_i.inputs
    labels_i = model_configuration_i.labels

    # Get all parameters of the configuration for domain j
    model_j = model_configuration_j.model
    inputs_j = model_configuration_j.inputs
    labels_j = model_configuration_j.labels

    # Set VAE models to eval for the training of the discriminator
    model_i.eval()
    model_j.eval()
    model_i.zero_grad()
    model_j.zero_grad()

    # Send models and data to device
    inputs_i, inputs_j = Variable(inputs_i).to(device), Variable(inputs_j).to(device)

    model_i.to(device)
    model_j.to(device)
    latent_dcm.to(device)
    latent_dcm_optimizer.zero_grad()

    # Set latent discriminator to train
    if phase == "train" and latent_dcm.trainable:
        latent_dcm.train()
    else:
        latent_dcm.eval()

    # Forward pass
    latents_i = Variable(model_i(inputs_i)["latents"])
    latents_j = Variable(model_j(inputs_j)["latents"])

    # add noise to inputs
    noise_i = torch.normal(0, 0.1, latents_i.size()).to(latents_i.device)
    noise_j = torch.normal(0, 0.1, latents_j.size()).to(latents_j.device)

    latents_i += noise_i
    latents_j += noise_j

    if use_latent_discriminator:
        labels_i, labels_j = (
            Variable(labels_i).to(device),
            Variable(labels_j).to(device),
        )

        # Add class label to latent representations to ensure that latent representations encode generic information
        # independent from the used data modality (see Adversarial AutoEncoder paper)

        dcm_input_i = torch.cat(
            (latents_i, labels_i.float().view(-1, 1).expand(-1, 20)), dim=1
        )
        dcm_input_j = torch.cat(
            (latents_j, labels_j.float().view(-1, 1).expand(-1, 20)), dim=1
        )

    else:
        dcm_input_i = latents_i
        dcm_input_j = latents_j

    dcm_output_i = latent_dcm(dcm_input_i)
    dcm_output_j = latent_dcm(dcm_input_j)

    domain_labels_i = torch.zeros(dcm_output_i.size(0)).long().to(device)
    domain_labels_j = torch.ones(dcm_output_j.size(0)).long().to(device)

    # Randomly flip 10% of the labels
    flip_idc = np.random.randint(0, latents_i.size()[0], int(0.1 * latents_i.size()[0]))
    domain_labels_i[flip_idc] = 1
    domain_labels_j[flip_idc] = 0

    # domain_labels_i = (
    #     torch.ones(dcm_output_i.size(0)).long().to(device).view(-1, 1) * 0.1
    # )
    # domain_labels_j = (
    #     torch.ones(dcm_output_j.size(0)).long().to(device).view(-1, 1) * 0.9
    # )

    dcm_loss = 0.5 * (
        latent_dcm_loss(dcm_output_i, domain_labels_i)
        + latent_dcm_loss(dcm_output_j, domain_labels_j)
    )

    # Backpropagate loss and update parameters if in phase 'train'
    if phase == "train" and latent_dcm.trainable:
        dcm_loss.backward()
        latent_dcm_optimizer.step()
    else:
        pass

    # Get summary statistics
    batch_size_i = inputs_i.size(0)
    batch_size_j = inputs_j.size(0)

    accuracy_i = accuracy(dcm_output_i, domain_labels_i)
    accuracy_j = accuracy(dcm_output_j, domain_labels_j)

    summary_stats = {
        "dcm_loss": dcm_loss.item() * (batch_size_i + batch_size_j),
        "accuracy_i": accuracy_i,
        "accuracy_j": accuracy_j,
    }

    return summary_stats


def process_epoch_two_domains(
    domain_configs: List[DomainConfig],
    latent_dcm: Module,
    latent_dcm_optimizer: Optimizer,
    latent_dcm_loss: Module,
    latent_structure_model: Module = None,
    latent_structure_model_optimizer: Optimizer = None,
    latent_structure_model_loss: Module = None,
    latent_distance_loss: Module = None,
    alpha: float = 0.1,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    lamb: float = 0.000001,
    use_latent_discriminator: bool = True,
    use_latent_structure_model: bool = False,
    phase: str = "train",
    device: str = "cuda:0",
) -> dict:
    # Get domain configurations for the two domains
    domain_config_i = domain_configs[0]
    domain_model_config_i = domain_config_i.domain_model_config
    data_loader_dict_i = domain_config_i.data_loader_dict
    train_loader_i = data_loader_dict_i[phase]
    data_key_i = domain_config_i.data_key
    label_key_i = domain_config_i.label_key

    domain_config_j = domain_configs[1]
    domain_model_config_j = domain_config_j.domain_model_config
    data_loader_dict_j = domain_config_j.data_loader_dict
    train_loader_j = data_loader_dict_j[phase]
    data_key_j = domain_config_j.data_key
    label_key_j = domain_config_j.label_key

    # Initialize epoch statistics
    recon_loss_i = 0
    recon_loss_j = 0
    dcm_loss = 0
    latent_sm_loss = 0
    ae_dcm_loss = 0
    kl_loss = 0
    distance_loss = 0
    total_loss = 0

    correct_preds_i = 0
    n_preds_i = 0
    correct_preds_j = 0
    n_preds_j = 0
    paired_distance_samples = 0

    n_class_preds_i = 0
    correct_class_preds_i = 0
    n_class_preds_j = 0
    correct_class_preds_j = 0

    if (
        domain_model_config_i.model.model_base_type
        != domain_model_config_i.model.model_base_type
    ):
        raise RuntimeError(
            "Model type mismatch: Got ({}, {}), Expected: matching model types".format(
                domain_model_config_i.model.model_base_type.upper(),
                domain_model_config_j.model.model_base_type.upper(),
            )
        )
    # Todo - continue case separation implementation for loss functions based on model base type
    model_base_type = domain_model_config_i.model.model_base_type.lower()
    # vae_mode = domain_model_config_i.model.model_base_type.lower() == "VAE"

    # partly_integrated_latent_space = domain_model_config_i.model.n_latent_spaces == 2

    # Iterate over batches
    for index, (samples_i, samples_j) in enumerate(zip(train_loader_i, train_loader_j)):
        # Set model_configs
        domain_model_config_i.inputs = samples_i[data_key_i]
        domain_model_config_i.labels = samples_i[label_key_i]

        domain_model_config_j.inputs = samples_j[data_key_j]
        domain_model_config_j.labels = samples_j[label_key_j]

        if "train_pair" in samples_i and "train_pair" in samples_j:
            paired_training_mask = samples_i["train_pair"]
            if not torch.all(torch.eq(paired_training_mask, samples_j["train_pair"])):
                raise RuntimeError("Samples seemed to be not aligned!")
        else:
            paired_training_mask = None

        domain_model_configs = [domain_model_config_i, domain_model_config_j]

        ae_train_summary = train_autoencoders_two_domains(
            domain_model_configurations=domain_model_configs,
            latent_dcm=latent_dcm,
            latent_dcm_loss=latent_dcm_loss,
            latent_structure_model=latent_structure_model,
            latent_structure_model_loss=latent_structure_model_loss,
            latent_structure_model_optimizer=latent_structure_model_optimizer,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            lamb=lamb,
            use_latent_discriminator=use_latent_discriminator,
            use_latent_structure_model=use_latent_structure_model,
            phase=phase,
            device=device,
            model_base_type=model_base_type,
            latent_distance_loss=latent_distance_loss,
            paired_training_mask=paired_training_mask,
        )
        # Update statistics after training the AE
        recon_loss_i += ae_train_summary["recon_loss_i"]
        recon_loss_j += ae_train_summary["recon_loss_j"]
        ae_dcm_loss += ae_train_summary["dcm_loss"]
        total_loss += ae_train_summary["total_loss"]

        if (
            paired_training_mask is not None
            and "latent_distance_loss" in ae_train_summary
        ):
            distance_loss += ae_train_summary["latent_distance_loss"]
            paired_distance_samples += ae_train_summary["paired_distance_samples"]

        if model_base_type == "vae":
            kl_loss += ae_train_summary["kl_loss"]
        elif model_base_type == "gmvae":
            kl_loss += ae_train_summary["kl_loss"]

        if use_latent_structure_model:
            latent_sm_loss += ae_train_summary["latent_structure_model_loss"]
            correct_class_preds_i += ae_train_summary[
                "latent_structure_model_accuracy_i"
            ][0]
            n_class_preds_i += ae_train_summary["latent_structure_model_accuracy_i"][1]
            correct_class_preds_j += ae_train_summary[
                "latent_structure_model_accuracy_j"
            ][0]
            n_class_preds_j += ae_train_summary["latent_structure_model_accuracy_j"][1]

        dcm_train_summary = train_latent_dcm_two_domains(
            domain_model_configurations=domain_model_configs,
            latent_dcm=latent_dcm,
            latent_dcm_loss=latent_dcm_loss,
            latent_dcm_optimizer=latent_dcm_optimizer,
            use_latent_discriminator=use_latent_discriminator,
            phase=phase,
            device=device,
        )

        # Update statistics after training the DCM:
        dcm_loss += dcm_train_summary["dcm_loss"]
        correct_preds_i += dcm_train_summary["accuracy_i"][0]
        n_preds_i += dcm_train_summary["accuracy_i"][1]
        correct_preds_j += dcm_train_summary["accuracy_j"][0]
        n_preds_j += dcm_train_summary["accuracy_j"][1]

    # Get average over batches for statistics
    recon_loss_i /= n_preds_i
    recon_loss_j /= n_preds_j

    dcm_loss /= n_preds_i + n_preds_j
    ae_dcm_loss /= n_preds_i + n_preds_j

    kl_loss /= n_preds_i + n_preds_j

    total_loss /= n_preds_i + n_preds_j

    accuracy_i = correct_preds_i / n_preds_i
    accuracy_j = correct_preds_j / n_preds_j

    epoch_statistics = {
        "recon_loss_i": recon_loss_i,
        "recon_loss_j": recon_loss_j,
        "dcm_loss": dcm_loss,
        "ae_dcm_loss": ae_dcm_loss,
        "accuracy_i": accuracy_i,
        "accuracy_j": accuracy_j,
        "total_loss": total_loss,
    }

    if model_base_type in ["vae", "gmvae"]:
        epoch_statistics["kl_loss"] = kl_loss

    if use_latent_structure_model:
        latent_sm_loss /= n_preds_i + n_preds_j
        class_i_accuracy = correct_class_preds_i / n_class_preds_i
        class_j_accuracy = correct_class_preds_j / n_class_preds_j
        epoch_statistics["latent_structure_model_loss"] = latent_sm_loss
        epoch_statistics["latent_structure_model_accuracy_i"] = class_i_accuracy
        epoch_statistics["latent_structure_model_accuracy_j"] = class_j_accuracy

    if paired_training_mask is not None and paired_distance_samples > 0:
        epoch_statistics["latent_distance_loss"] = (
            distance_loss / paired_distance_samples
        )

    return epoch_statistics


def train_val_test_loop_two_domains(
    output_dir: str,
    domain_configs: List[DomainConfig],
    latent_dcm_config: dict = None,
    latent_dcm_train_freq: int = 1,
    latent_structure_model_config: dict = None,
    alpha: float = 0.1,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    lamb: float = 0.00001,
    use_latent_discriminator: bool = True,
    use_latent_structure_model: bool = False,
    num_epochs: int = 500,
    save_freq: int = 10,
    early_stopping: int = 20,
    device: str = None,
    paired_mode: bool = False,
    latent_distance_loss: Module = None,
) -> Tuple[dict, dict]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get available device, if cuda is available the GPU will be used
    if not device:
        device = get_device()

    # Store start time of the training
    start_time = time.time()

    # Initialize early stopping counter
    es_counter = 0
    if early_stopping < 0:
        early_stopping = num_epochs

    total_loss_dict = {"train": [], "val": []}

    epoch_with_best_model = 0

    # Reserve space to save best model configurations
    domain_names = [domain_configs[0].name, domain_configs[1].name]
    best_model_i_weights = (
        domain_configs[0].domain_model_config.model.cpu().state_dict()
    )
    best_model_j_weights = (
        domain_configs[1].domain_model_config.model.cpu().state_dict()
    )
    best_model_configs = {
        "model_i_weights": best_model_i_weights,
        "model_j_weights": best_model_j_weights,
    }

    for i in range(len(domain_names)):
        logging.debug("Model for domain {}:".format(domain_names[i]))
        logging.debug(domain_configs[i].domain_model_config.model)

    # Unpack latent model configurations
    if latent_dcm_config is not None:
        latent_dcm = latent_dcm_config["model"]
        logging.debug("Latent discriminator:")
        logging.debug(latent_dcm)
        latent_dcm_optimizer = latent_dcm_config["optimizer"]
        latent_dcm_loss = latent_dcm_config["loss"]
    else:
        latent_dcm = None
        latent_dcm_optimizer = None
        latent_dcm_loss = None

    if latent_structure_model_config is not None:
        latent_structure_model = latent_structure_model_config["model"]
        logging.debug("Latent structure model")
        logging.debug(latent_structure_model)
        latent_structure_model_optimizer = latent_structure_model_config["optimizer"]
        latent_structure_model_loss = latent_structure_model_config["loss"]
    else:
        latent_structure_model = None
        latent_structure_model_optimizer = None
        latent_structure_model_loss = None

    if latent_dcm is not None:
        best_latent_dcm_weights = latent_dcm.state_dict()
        best_model_configs["dcm_weights"] = best_latent_dcm_weights

    if latent_structure_model is not None:
        best_latent_structure_model_weights = latent_structure_model.state_dict()
        best_model_configs[
            "latent_structure_model_weights"
        ] = best_latent_structure_model_weights

    # Initialize current best loss
    best_total_loss = np.infty

    # Iterate over the epochs
    for i in range(num_epochs):
        logging.debug("---" * 20)
        logging.debug("---" * 20)
        logging.debug("Started epoch {}/{}".format(i + 1, num_epochs))
        logging.debug("---" * 20)

        # Check if early stopping is triggered
        if es_counter > early_stopping:
            logging.debug(
                "Training was stopped early due to no improvement of the validation"
                " loss for {} epochs.".format(early_stopping)
            )
            break

        # Iterate over training and validation phase
        for phase in ["train", "val"]:
            # if latent_dcm is not None:
            # if i % 1 <=0 :
            #     latent_dcm.trainable = True
            #     #domain_configs[0].domain_model_config.model.trainable = False
            #     domain_configs[1].domain_model_config.model.trainable = False
            # else:
            #     latent_dcm.trainable = False
            #     #domain_configs[0].domain_model_config.model.trainable = True
            #     domain_configs[1].domain_model_config.model.trainable = True

            epoch_statistics = process_epoch_two_domains(
                domain_configs=domain_configs,
                latent_dcm=latent_dcm,
                latent_dcm_optimizer=latent_dcm_optimizer,
                latent_dcm_loss=latent_dcm_loss,
                latent_structure_model=latent_structure_model,
                latent_structure_model_optimizer=latent_structure_model_optimizer,
                latent_structure_model_loss=latent_structure_model_loss,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                lamb=lamb,
                use_latent_discriminator=use_latent_discriminator,
                use_latent_structure_model=use_latent_structure_model,
                phase=phase,
                device=device,
                latent_distance_loss=latent_distance_loss,
            )

            logging.debug(
                "{} LOSS STATISTICS FOR EPOCH {}: ".format(phase.upper(), i + 1)
            )

            logging.debug(
                "Reconstruction Loss for {} domain: {:.8f}".format(
                    domain_names[0], epoch_statistics["recon_loss_i"]
                )
            )
            logging.debug(
                "Accuracy of DCM for {} domain: {:.8f}".format(
                    domain_names[0], epoch_statistics["accuracy_i"]
                )
            )

            logging.debug(
                "Reconstruction Loss for {} domain: {:.8f}".format(
                    domain_names[1], epoch_statistics["recon_loss_j"]
                )
            )
            logging.debug(
                "Accuracy of DCM for {} domain: {:.8f}".format(
                    domain_names[1], epoch_statistics["accuracy_j"]
                )
            )

            logging.debug(
                "Latent autoencoder discriminator loss: {:.8f}".format(
                    epoch_statistics["ae_dcm_loss"]
                )
            )
            logging.debug(
                "Latent discriminator loss: {:.8f}".format(epoch_statistics["dcm_loss"])
            )
            if "kl_loss" in epoch_statistics:
                logging.debug(
                    "Latent kld regularizer loss: {:.8f}".format(
                        epoch_statistics["kl_loss"]
                    )
                )

            if "latent_structure_model_loss" in epoch_statistics:
                logging.debug(
                    "Latent structure model loss: {:.8f}".format(
                        epoch_statistics["latent_structure_model_loss"]
                    )
                )
                logging.debug(
                    "Latent structure model accuracy for {} domain: {:.8f}".format(
                        domain_names[0],
                        epoch_statistics["latent_structure_model_accuracy_i"],
                    )
                )
                logging.debug(
                    "Latent structure model accuracy for {} domain: {:.8f}".format(
                        domain_names[1],
                        epoch_statistics["latent_structure_model_accuracy_j"],
                    )
                )

            if "latent_distance_loss" in epoch_statistics:
                logging.debug(
                    "Latent distance loss: {:.8f}".format(
                        epoch_statistics["latent_distance_loss"]
                    )
                )

            if phase == "val" and paired_mode:
                metrics = evaluate_latent_integration(
                    model_i=domain_configs[0].domain_model_config.model,
                    model_j=domain_configs[1].domain_model_config.model,
                    data_loader_i=domain_configs[0].data_loader_dict["val"],
                    data_loader_j=domain_configs[1].data_loader_dict["val"],
                    device=device,
                )
                logging.debug(
                    "Latent l1 distance of paired data normalized by the mean l1"
                    " distance of the unpaired data: {:.8f}".format(
                        metrics["latent_l1_distance"]
                    )
                )
                for k, v in metrics["knn_accs"].items():
                    logging.debug(
                        "{}-NN accuracy for the paired data: {:.8f}".format(k, v)
                    )

            logging.debug("***" * 20)

            epoch_total_loss = epoch_statistics["total_loss"]
            logging.debug("Total {} loss: {:.8f}".format(phase, epoch_total_loss))
            logging.debug("***" * 20)

            total_loss_dict[phase].append(epoch_total_loss)

            if i % save_freq == 0:

                domain_model_configs = [
                    domain_configs[0].domain_model_config,
                    domain_configs[1].domain_model_config,
                ]
                if domain_names[0] == "image":
                    visualize_image_translation_performance(
                        domain_model_configs=domain_model_configs,
                        epoch=i,
                        output_dir=output_dir,
                        device=device,
                        phase=phase,
                    )

                # Save model states regularly
                checkpoint_dir = "{}/epoch_{}".format(output_dir, i)
                visualize_model_performance(
                    output_dir=checkpoint_dir,
                    domain_configs=domain_configs,
                    dataset_types=[phase],
                    device=device,
                    reduction="umap",
                )
                visualize_model_performance(
                    output_dir=checkpoint_dir,
                    domain_configs=domain_configs,
                    dataset_types=[phase],
                    device=device,
                    reduction="tsne",
                )
                visualize_model_performance(
                    output_dir=checkpoint_dir,
                    domain_configs=domain_configs,
                    dataset_types=[phase],
                    device=device,
                    reduction="pca",
                )

                model_i_weights = (
                    domain_configs[0].domain_model_config.model.cpu().state_dict()
                )
                model_j_weights = (
                    domain_configs[1].domain_model_config.model.cpu().state_dict()
                )
                torch.save(
                    model_i_weights,
                    "{}/model_{}.pth".format(checkpoint_dir, domain_names[0]),
                )
                torch.save(
                    model_j_weights,
                    "{}/model_{}.pth".format(checkpoint_dir, domain_names[1]),
                )

                if latent_dcm is not None:
                    latent_dcm_weights = latent_dcm.cpu().state_dict()
                    torch.save(latent_dcm_weights, "{}/dcm.pth".format(checkpoint_dir))
                if latent_structure_model is not None:
                    latent_structure_model_weights = (
                        latent_structure_model.cpu().state_dict()
                    )
                    torch.save(
                        latent_structure_model_weights,
                        "{}/latent_structure_model.pth".format(checkpoint_dir),
                    )

    # Training complete
    time_elapsed = time.time() - start_time

    logging.debug("###" * 20)
    logging.debug(
        "Training completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, int(time_elapsed % 60)
        )
    )

    # Load best models
    # logging.debug(
    #     "Load best model configurations found in epoch {}.".format(
    #         epoch_with_best_model
    #     )
    # )
    # Todo removed loading of best model - because for adversarial training the total loss is too unstable for
    # this model selection procedure to work.

    # domain_configs[0].domain_model_config.model.load_state_dict(best_model_i_weights)
    # domain_configs[1].domain_model_config.model.load_state_dict(best_model_j_weights)
    #
    # if latent_dcm is not None:
    #     latent_dcm.load_state_dict(best_latent_dcm_weights)
    # if latent_structure_model is not None:
    #     latent_structure_model.load_state_dict(best_latent_structure_model_weights)

    if "test" in domain_configs[0].data_loader_dict:
        epoch_statistics = process_epoch_two_domains(
            domain_configs=domain_configs,
            latent_dcm=latent_dcm,
            latent_dcm_optimizer=latent_dcm_optimizer,
            latent_dcm_loss=latent_dcm_loss,
            latent_structure_model=latent_structure_model,
            latent_structure_model_optimizer=latent_structure_model_optimizer,
            latent_structure_model_loss=latent_structure_model_loss,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lamb=lamb,
            use_latent_discriminator=use_latent_discriminator,
            use_latent_structure_model=use_latent_structure_model,
            phase="test",
            device=device,
            latent_distance_loss=latent_distance_loss,
        )

        logging.debug("###" * 20)
        logging.debug("TEST LOSS STATISTICS: ")

        logging.debug(
            "Reconstruction Loss for {} domain: {:.8f}".format(
                domain_names[0], epoch_statistics["recon_loss_i"]
            )
        )
        logging.debug(
            "Accuracy of DCM for {} domain: {:.8f}".format(
                domain_names[0], epoch_statistics["accuracy_i"]
            )
        )

        logging.debug(
            "Reconstruction Loss for {} domain: {:.8f}".format(
                domain_names[1], epoch_statistics["recon_loss_j"]
            )
        )
        logging.debug(
            "Accuracy of DCM for {} domain: {:.8f}".format(
                domain_names[1], epoch_statistics["accuracy_j"]
            )
        )

        logging.debug(
            "Latent autoencoder discriminator loss: {:.8f}".format(
                epoch_statistics["ae_dcm_loss"]
            )
        )
        logging.debug(
            "Latent discriminator loss: {:.8f}".format(epoch_statistics["dcm_loss"])
        )
        if "kl_loss" in epoch_statistics:
            logging.debug(
                "Latent kld regularizer loss: {:.8f}".format(
                    epoch_statistics["kl_loss"]
                )
            )
        if "latent_structure_model_loss" in epoch_statistics:
            logging.debug(
                "Latent structure model loss: {:.8f}".format(
                    epoch_statistics["latent_structure_model_loss"]
                )
            )
            logging.debug(
                "Latent structure model accuracy for domain {}: {:.8f}".format(
                    domain_names[0],
                    epoch_statistics["latent_structure_model_accuracy_i"],
                )
            )
            logging.debug(
                "Latent structure model accuracy for domain {}: {:.8f}".format(
                    domain_names[1],
                    epoch_statistics["latent_structure_model_accuracy_j"],
                )
            )

        if "latent_distance_loss" in epoch_statistics:
            logging.debug(
                "Latent distance loss: {:.8f}".format(
                    epoch_statistics["latent_distance_loss"]
                )
            )

        if paired_mode:
            metrics = evaluate_latent_integration(
                model_i=domain_configs[0].domain_model_config.model,
                model_j=domain_configs[1].domain_model_config.model,
                data_loader_i=domain_configs[0].data_loader_dict["test"],
                data_loader_j=domain_configs[1].data_loader_dict["test"],
                device=device,
            )
            logging.debug(
                "Latent l1 distance of paired data normalized by the mean l1 distance"
                " of the unpaired data: {:.8f}".format(metrics["latent_l1_distance"])
            )
            for k, v in metrics["knn_accs"].items():
                logging.debug("{}-NN accuracy for the paired data: {:.8f}".format(k, v))

        logging.debug("***" * 20)

        epoch_total_loss = epoch_statistics["total_loss"]
        logging.debug("Total test loss: {:.8f}".format(epoch_total_loss))

        logging.debug("***" * 20)

    # Summarize return parameters and save final models
    trained_models = {
        "model_i": domain_configs[0].domain_model_config.model,
        "model_j": domain_configs[1].domain_model_config.model,
    }

    torch.save(
        domain_configs[0].domain_model_config.model.state_dict(),
        "{}/final_{}_model.pth".format(output_dir, domain_names[0]),
    )
    torch.save(
        domain_configs[1].domain_model_config.model.state_dict(),
        "{}/final_{}_model.pth".format(output_dir, domain_names[1]),
    )

    if latent_dcm is not None:
        trained_models["dcm"] = latent_dcm
        torch.save(latent_dcm.state_dict(), "{}/final_dcm.pth".format(output_dir))

    if latent_structure_model is not None:
        trained_models["latent_structure_model"] = latent_structure_model
        torch.save(
            latent_structure_model.state_dict(),
            "{}/final_latent_structure_model.pth".format(output_dir),
        )

    # Visualize results
    test_dir = "{}/test".format(output_dir)
    os.makedirs(test_dir, exist_ok=True)
    visualize_model_performance(
        output_dir=test_dir,
        domain_configs=domain_configs,
        dataset_types=["train", "val", "test"],
        device=device,
    )

    return trained_models, total_loss_dict


def train_autoencoder(
    domain_model_config: DomainModelConfig,
    latent_structure_model: Module,
    latent_structure_model_optimizer: Optimizer,
    latent_structure_model_loss: Module,
    alpha: float = 1.0,
    gamma: float = 0.001,
    lamb: float = 1e-8,
    phase: str = "train",
    use_latent_structure_model: bool = True,
    device: str = "cuda:0",
    model_base_type: str = "ae",
) -> dict:
    # Get all parameters of the configuration for domain i
    model = domain_model_config.model
    optimizer = domain_model_config.optimizer
    inputs = domain_model_config.inputs
    labels = domain_model_config.labels
    recon_loss_fct = domain_model_config.recon_loss_function
    train = domain_model_config.trainable

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    # Set V/AE model to train if defined in respective configuration
    model.to(device)

    if phase == "train" and train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    if use_latent_structure_model:
        assert latent_structure_model is not None
        if phase == "train":
            latent_structure_model.train()
        else:
            latent_structure_model.eval()
        latent_structure_model.to(device)
        latent_structure_model_optimizer.zero_grad()

    # Forward pass of the V/AE
    inputs = Variable(inputs).to(device)
    labels = Variable(labels).to(device)

    outputs = model(inputs)
    recons = outputs["recons"]
    latents = outputs["latents"]

    if model_base_type == "vae":
        mu = outputs["mu"]

        logvar = outputs["logvar"]

        loss_dict = model.loss_function(
            inputs=inputs, recons=recons, mu=mu, logvar=logvar
        )

        recon_loss = loss_dict["recon_loss"]
        kld_loss = loss_dict["kld_loss"]

        kl_loss = kld_loss
        total_loss = recon_loss * alpha + kl_loss * lamb
    elif model_base_type == "gmvae":
        logits = outputs["logits"]
        probs = outputs["probs"]
        component_labels = outputs["component_labels"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        mu_component_prior = outputs["mu_component_prior"]
        logvar_component_prior = outputs["logvar_component_prior"]

        loss_dict = model.loss_function(
            inputs=inputs,
            recons=recons,
            mu=mu,
            logvar=logvar,
            mu_prior=mu_component_prior,
            logvar_prior=logvar_component_prior,
            y_probs=probs,
            y_logits=logits,
            y_true=labels,
        )

        recon_loss = loss_dict["recon_loss"]
        kld_loss = loss_dict["kld_loss"]
        component_prior_loss = loss_dict["component_prior_loss"]
        component_supervision_loss = loss_dict["component_supervision_loss"]

        kl_loss = kld_loss

        total_loss = recon_loss * alpha + lamb * kl_loss
        if component_prior_loss is not None:
            total_loss += component_prior_loss * gamma
        elif component_supervision_loss is not None:
            total_loss += component_supervision_loss * gamma

    elif model_base_type == "ae":
        recon_loss = model.loss_function(inputs=inputs, recons=recons)["recon_loss"]
        total_loss = recon_loss
    else:
        raise RuntimeError("Unknown model type: {}".format(model_base_type))

    # Forward pass latent structure model if it is supposed to be trained and used to assess the integration of the
    # learned latent spaces
    if use_latent_structure_model:
        latent_structure_model_output = latent_structure_model(latents)

    # # Add loss of latent structure model if this is trained
    if use_latent_structure_model:
        latent_sm_loss = latent_structure_model_loss(
            latent_structure_model_output, labels.view(-1).long()
        )
        total_loss += latent_sm_loss * gamma

    # Backpropagate loss and update parameters if we are in the training phase
    if phase == "train":
        total_loss.backward()
        if train:
            optimizer.step()
            model.updated = True
        if use_latent_structure_model and latent_structure_model.trainable:
            latent_structure_model_optimizer.step()

        scheduler.step(total_loss)

    # Get summary statistics
    batch_size = inputs.size(0)
    total_loss_item = recon_loss.item() * batch_size * alpha

    batch_statistics = {"recon_loss": recon_loss.item() * batch_size}

    if model_base_type == "vae":
        batch_statistics["kl_loss"] = kl_loss.item()
        total_loss_item += kl_loss.item() * lamb

    if use_latent_structure_model:
        batch_statistics["latent_structure_model_loss"] = (
            latent_sm_loss.item() * batch_size
        )
        batch_statistics["accuracy"] = accuracy(latent_structure_model_output, labels)
        total_loss_item += latent_sm_loss.item() * batch_size * gamma

    batch_statistics["total_loss"] = total_loss_item

    return batch_statistics


def process_epoch_single_domain(
    domain_config: DomainConfig,
    latent_structure_model: Module = None,
    latent_structure_model_optimizer: Optimizer = None,
    latent_structure_model_loss: Module = None,
    alpha: float = 1.0,
    gamma: float = 0.001,
    lamb: float = 1e-8,
    use_latent_structure_model: bool = True,
    phase: str = "train",
    device: str = "cuda:0",
) -> dict:
    # Get domain configurations for the domain
    domain_model_config = domain_config.domain_model_config
    data_loader_dict = domain_config.data_loader_dict
    data_loader = data_loader_dict[phase]
    data_key = domain_config.data_key
    label_key = domain_config.label_key

    # Initialize epoch statistics
    recon_loss = 0
    latent_sm_loss = 0
    kl_loss = 0
    total_loss = 0

    correct_preds = 0
    n_preds = 0

    model_base_type = domain_model_config.model.model_base_type.lower()

    # Iterate over batches
    for index, samples in enumerate(data_loader):
        # Set model_configs
        domain_model_config.inputs = samples[data_key]
        domain_model_config.labels = samples[label_key]

        batch_statistics = train_autoencoder(
            domain_model_config=domain_model_config,
            latent_structure_model=latent_structure_model,
            latent_structure_model_optimizer=latent_structure_model_optimizer,
            latent_structure_model_loss=latent_structure_model_loss,
            alpha=alpha,
            gamma=gamma,
            lamb=lamb,
            phase=phase,
            device=device,
            use_latent_structure_model=use_latent_structure_model,
            model_base_type=model_base_type,
        )

        recon_loss += batch_statistics["recon_loss"]
        if use_latent_structure_model:
            latent_sm_loss += batch_statistics["latent_structure_model_loss"]
            correct_preds += batch_statistics["accuracy"][0]
            n_preds += batch_statistics["accuracy"][1]
        if model_base_type == "vae":
            kl_loss += batch_statistics["kl_loss"]
        total_loss += batch_statistics["total_loss"]

    # Get average over batches for statistics
    recon_loss /= len(data_loader.dataset)
    kl_loss /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)
    if use_latent_structure_model:
        accuracy = correct_preds / len(data_loader.dataset)
    else:
        accuracy = 0

    epoch_statistics = {
        "recon_loss": recon_loss,
        "accuracy": accuracy,
        "total_loss": total_loss,
    }
    if model_base_type == "vae":
        epoch_statistics["kl_loss"] = kl_loss

    if use_latent_structure_model:
        latent_sm_loss /= n_preds
        epoch_statistics["latent_structure_model_loss"] = latent_sm_loss

    return epoch_statistics


def train_val_test_loop_single_domain(
    output_dir: str,
    domain_config: DomainConfig,
    latent_structure_model_config: dict = None,
    alpha: float = 1.0,
    gamma: float = 0.001,
    lamb: float = 0.0000001,
    use_latent_structure_model: bool = False,
    num_epochs: int = 500,
    save_freq: int = 10,
    early_stopping: int = 20,
    device: str = None,
) -> Tuple[dict, dict]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get available device, if cuda is available the GPU will be used
    if not device:
        device = get_device()

    # Store start time of the training
    start_time = time.time()

    # Initialize early stopping counter
    es_counter = 0
    if early_stopping < 0:
        early_stopping = num_epochs

    total_loss_dict = {"train": [], "val": []}

    # Reserve space for best model weights
    best_model_weights = domain_config.domain_model_config.model.cpu().state_dict()

    if use_latent_structure_model:
        latent_structure_model = latent_structure_model_config["model"]
        latent_structure_model_optimizer = latent_structure_model_config["optimizer"]
        latent_structure_model_loss = latent_structure_model_config["loss"]
    else:
        latent_structure_model = None
        latent_structure_model_optimizer = None
        latent_structure_model_loss = None

    # Reserve space for best latent latent_structure_model weights
    if latent_structure_model is not None:
        best_latent_structure_model_weights = latent_structure_model.cpu().state_dict()
    else:
        best_latent_structure_model_weights = None

    best_model_configs = {
        "best_model": best_model_weights,
        "best_latent_structure_model_weights": best_latent_structure_model_weights,
    }

    best_total_loss = np.infty

    # Iterate over the epochs
    for i in range(num_epochs):
        logging.debug("---" * 20)
        logging.debug("---" * 20)
        logging.debug("Started epoch {}/{}".format(i + 1, num_epochs))
        logging.debug("---" * 20)

        # Check if early stopping is triggered
        if es_counter > early_stopping:
            logging.debug(
                "Training was stopped early due to no improvement of the validation"
                " loss for {} epochs.".format(early_stopping)
            )
            break

        # Iterate over training and validation phase
        for phase in ["train", "val"]:
            epoch_statistics = process_epoch_single_domain(
                domain_config=domain_config,
                latent_structure_model=latent_structure_model,
                latent_structure_model_optimizer=latent_structure_model_optimizer,
                latent_structure_model_loss=latent_structure_model_loss,
                alpha=alpha,
                gamma=gamma,
                lamb=lamb,
                use_latent_structure_model=use_latent_structure_model,
                phase=phase,
                device=device,
            )

            logging.debug(
                "{} LOSS STATISTICS FOR EPOCH {}: ".format(phase.upper(), i + 1)
            )

            logging.debug(
                "Reconstruction loss for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["recon_loss"]
                )
            )
            if "kl_loss" in epoch_statistics:
                logging.debug(
                    "KL loss for {} domain: {:.8f}".format(
                        domain_config.name, epoch_statistics["kl_loss"]
                    )
                )

            if use_latent_structure_model:
                logging.debug(
                    "Latent structure model loss for {} domain: {:.8f}".format(
                        domain_config.name,
                        epoch_statistics["latent_structure_model_loss"],
                    )
                )

                logging.debug(
                    "Latent structure model accuracy for {} domain: {:.8f}".format(
                        domain_config.name, epoch_statistics["accuracy"]
                    )
                )

            epoch_total_loss = epoch_statistics["total_loss"]
            total_loss_dict[phase].append(epoch_total_loss)
            logging.debug("***" * 20)
            logging.debug(
                "Total {} loss for {} domain: {:.8f}".format(
                    phase, domain_config.name, epoch_total_loss
                )
            )
            logging.debug("***" * 20)

            if i % save_freq == 0 and domain_config.name == "image":
                if domain_config.domain_model_config.model.model_base_type == "vae":
                    visualize_image_vae_performance(
                        domain_model_config=domain_config.domain_model_config,
                        epoch=i,
                        output_dir=output_dir,
                        device=device,
                        phase=phase,
                    )
                else:
                    visualize_image_ae_performance(
                        domain_model_config=domain_config.domain_model_config,
                        epoch=i,
                        output_dir=output_dir,
                        device=device,
                        phase=phase,
                    )

            if phase == "val":
                # Save model states if current parameters give the best validation loss
                if epoch_total_loss < best_total_loss:
                    es_counter = 0
                    best_total_loss = epoch_total_loss

                    best_model_weights = copy.deepcopy(
                        domain_config.domain_model_config.model.cpu().state_dict()
                    )
                    best_model_configs["best_model_weights"] = best_model_weights

                    torch.save(
                        best_model_weights, "{}/best_model.pth".format(output_dir),
                    )

                    if latent_structure_model is not None:
                        best_latent_structure_model_weights = copy.deepcopy(
                            latent_structure_model.cpu().state_dict()
                        )
                        best_model_configs[
                            "latent_structure_model_weights"
                        ] = best_latent_structure_model_weights
                        torch.save(
                            best_latent_structure_model_weights,
                            "{}/best_latent_structure_model.pth".format(output_dir),
                        )
                else:
                    es_counter += 1

                if i % save_freq == 0:

                    # Save model states regularly
                    checkpoint_dir = "{}/epoch_{}".format(output_dir, i)

                    visualize_model_performance(
                        output_dir=checkpoint_dir,
                        domain_configs=[domain_config],
                        dataset_types=["train", "val"],
                        device=device,
                    )

                    torch.save(
                        domain_config.domain_model_config.model.state_dict(),
                        "{}/model.pth".format(checkpoint_dir),
                    )
                    if use_latent_structure_model:
                        torch.save(
                            latent_structure_model.state_dict(),
                            "{}/latent_structure_model.pth".format(checkpoint_dir),
                        )

    # Training complete
    time_elapsed = time.time() - start_time

    logging.debug("###" * 20)
    logging.debug(
        "Training completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, int(time_elapsed % 60)
        )
    )

    # Load best model
    # domain_config.domain_model_config.model.load_state_dict(best_model_weights)
    # if latent_structure_model is not None:
    #    latent_structure_model.load_state_dict(best_latent_structure_model_weights)

    if "test" in domain_config.data_loader_dict:
        epoch_statistics = process_epoch_single_domain(
            domain_config=domain_config,
            latent_structure_model=latent_structure_model,
            latent_structure_model_optimizer=latent_structure_model_optimizer,
            latent_structure_model_loss=latent_structure_model_loss,
            gamma=gamma,
            lamb=lamb,
            use_latent_structure_model=use_latent_structure_model,
            phase="test",
            device=device,
        )

        logging.debug("TEST LOSS STATISTICS")

        logging.debug(
            "Reconstruction Loss for {} domain: {:.8f}".format(
                domain_config.name, epoch_statistics["recon_loss"]
            )
        )

        if "kl_loss" in epoch_statistics:
            logging.debug(
                "KL loss for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["kl_loss"]
                )
            )

        if use_latent_structure_model:
            logging.debug(
                "Latent structure model loss for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["latent_structure_model_loss"]
                )
            )
            logging.debug(
                "Latent structure model accuracy for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["accuracy"]
                )
            )
        logging.debug("***" * 20)
        logging.debug(
            "Total test loss for {} domain: {:.8f}".format(
                domain_config.name, epoch_statistics["total_loss"]
            )
        )
        logging.debug("***" * 20)

        test_dir = "{}/test".format(output_dir)
        os.makedirs(test_dir, exist_ok=True)

        # Visualize performance
        if domain_config.name == "image":
            if domain_config.domain_model_config.model.model_base_type == "vae":
                visualize_image_vae_performance(
                    domain_model_config=domain_config.domain_model_config,
                    epoch=i,
                    output_dir=output_dir,
                    device=device,
                    phase="test",
                )
            else:
                visualize_image_ae_performance(
                    domain_model_config=domain_config.domain_model_config,
                    epoch=i,
                    output_dir=output_dir,
                    device=device,
                    phase="test",
                )

        visualize_model_performance(
            output_dir=test_dir,
            domain_configs=[domain_config],
            dataset_types=["train", "val", "test"],
            device=device,
        )

    # Summarize return parameters
    fitted_models = {"model": domain_config.domain_model_config.model}
    if latent_structure_model is not None:
        fitted_models["latent_structure_model"] = latent_structure_model

    return fitted_models, total_loss_dict
