import copy
import logging
import os
import time
from typing import Tuple, List

import numpy as np
import torch
from torch import nn, device
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tnrange, tqdm_notebook

from src.functions.loss_functions import KLDLoss, kl_divergence
from src.functions.metrics import accuracy
from src.helper.models import DomainModelConfiguration
from src.utils.basic.visualization import visualize_model_performance
from src.utils.torch.general import get_device
from torch.nn import Module


def train_autoencoders_two_domains(
    domain_model_configurations: List[DomainModelConfiguration],
    latent_dcm: Module,
    latent_dcm_loss: Module,
    latent_clf: Module = None,
    latent_clf_optimizer: Optimizer = None,
    latent_clf_loss: Module = None,
    alpha: float = 1.0,
    device: str = "cuda:0",
    use_dcm: bool = True,
    use_clf: bool = True,
)->dict:

    # Expects 2 model configurations (one for each domain)
    model_configuration_i = domain_model_configurations[0]
    model_configuration_j = domain_model_configurations[1]

    # Get all parameters of the configuration for domain i
    vae_i = model_configuration_i.model
    optimizer_i = model_configuration_i.optimizer
    inputs_i = model_configuration_i.inputs
    labels_i = model_configuration_i.labels
    recon_loss_fct_i = model_configuration_i.recon_loss_function
    train_i = model_configuration_i.train

    # Get all parameters of the configuration for domain j
    vae_j = model_configuration_j.model
    optimizer_j = model_configuration_j.optimizer
    inputs_j = model_configuration_j.inputs
    labels_j = model_configuration_j.labels
    recon_loss_fct_j = model_configuration_j.recon_loss_function
    train_j = model_configuration_j.train

    # Set VAE models to train if defined in respective configuration
    if train_i:
        vae_i.train()
        vae_i.to(device)
        vae_i.zero_grad()
    if train_j:
        vae_j.train()
        vae_j.to(device)
        vae_j.zero_grad()

    # The discriminator will not be trained but only used to compute the adversarial loss for the AE updates
    latent_dcm.eval()
    latent_dcm.to(device)

    if use_clf:
        assert latent_clf is not None
        latent_clf.train()
        latent_clf.to(device)
        latent_clf.zero_grad()

    # Forward pass of the VAE
    inputs_i, inputs_j = Variable(inputs_i).to(device), Variable(inputs_j).to(device)
    recons_i, latents_i, mu_i, logvar_i = vae_i(inputs_i)
    recons_j, latents_j, mu_j, logvar_j = vae_j(inputs_j)

    if use_dcm:
        labels_i, labels_j = (
            Variable(labels_i).to(device),
            Variable(labels_j).to(device),
        )

        # Add class label to latent representations to ensure that latent representations encode generic information
        # independent from the used data modality (see Adversarial AutoEncoder paper)
        dcm_input_i = torch.cat(
            (latents_i, labels_i.float().view(-1, 1).expand(-1, 1)), dim=1
        )
        dcm_input_j = torch.cat(
            (latents_j, labels_j.float().view(-1, 1).expand(-1, 1)), dim=1
        )

    else:
        dcm_input_i = latents_i
        dcm_input_j = latents_j

    dcm_output_i = latent_dcm(dcm_input_i)
    dcm_output_j = latent_dcm(dcm_input_j)

    domain_labels_i = torch.zeros(dcm_output_i.size(0)).long()
    domain_labels_j = torch.ones(dcm_output_j.size(0)).long()

    # Forward pass latent classifier if it is supposed to be trained and used to assess the integration of the learned
    # latent spaces
    if use_clf:
        clf_output_i = latent_clf(latents_i)
        clf_output_j = latent_clf(latents_j)

    # Compute losses

    recon_loss_i = recon_loss_fct_i(recons_i, inputs_i)
    recon_loss_j = recon_loss_fct_j(recons_j, inputs_j)

    kl_loss = kl_divergence(mu_i, logvar_i) + kl_divergence(mu_j, logvar_j)

    # Calculate adversarial loss - by mixing labels indicating domain with output predictions to "confuse" the
    # discriminator and encourage learning autoencoder that make the distinction between the modalities in the latent
    # space as difficult as possible for the discriminator
    dcm_loss = 0.5 * latent_dcm_loss(
        dcm_output_i, domain_labels_j
    ) + 0.5 * latent_dcm_loss(dcm_output_j, domain_labels_i)

    total_loss = alpha * (recon_loss_i + recon_loss_j) + kl_loss + dcm_loss

    # Add loss of latent classifier if this is trained
    if use_clf:
        clf_loss = 0.5 * (
            latent_clf_loss(clf_output_i, labels_i)
            + latent_clf_loss(clf_output_j, labels_j)
        )
        total_loss += clf_loss

    # Backpropagate loss and update parameters
    total_loss.backward()
    if train_i:
        optimizer_i.step()
    if train_j:
        optimizer_j.step()
    if use_clf:
        latent_clf_optimizer.step()

    # Get summary statistics
    batch_size_i = inputs_i.size(0)
    batch_size_j = inputs_j.size(0)
    latent_size_i = mu_i.size(0)
    latent_size_j = mu_j.size(0)
    summary_stats = {
        "recon_loss_i": recon_loss_i * batch_size_i,
        "recon_loss_j": recon_loss_j * batch_size_j,
        "dcm_loss": dcm_loss * (batch_size_i + batch_size_j),
        "kl_loss": kl_loss * (latent_size_i + latent_size_j),
        "total_loss": total_loss,
    }
    return summary_stats


def train_latent_dcm_two_domains(domain_model_configurations:List[DomainModelConfiguration], latent_dcm:nn.Module, latent_dcm_optimizer:Optimizer, latent_dcm_loss:Module, use_dcm:bool,
                                 device:str='cuda:0'):

    # Get the model configurations for the two domains
    model_configuration_i = domain_model_configurations[0]
    model_configuration_j = domain_model_configurations[1]

    # Get all parameters of the configuration for domain i
    vae_i = model_configuration_i.model
    optimizer_i = model_configuration_i.optimizer
    inputs_i = model_configuration_i.inputs
    labels_i = model_configuration_i.labels
    recon_loss_fct_i = model_configuration_i.recon_loss_function

    # Get all parameters of the configuration for domain j
    vae_j = model_configuration_j.model
    optimizer_j = model_configuration_j.optimizer
    inputs_j = model_configuration_j.inputs
    labels_j = model_configuration_j.labels
    recon_loss_fct_j = model_configuration_j.recon_loss_function

    # Set VAE models to eval for the training of the discriminator
    vae_i.eval()
    vae_j.eval()

    # Send models and data to device
    inputs_i, inputs_j = Variable(inputs_i).to(device), Variable(inputs_j).to(device)

    vae_i.to(device)
    vae_j.to(device)

    # Reset gradients
    latent_dcm.zero_grad()

    # Forward pass
    _, latents_i, _, _ = vae_i(inputs_i)
    _, latents_j, _, _ = vae_j(inputs_j)

    if use_dcm:
        labels_i, labels_j = (
            Variable(labels_i).to(device),
            Variable(labels_j).to(device),
        )

        # Add class label to latent representations to ensure that latent representations encode generic information
        # independent from the used data modality (see Adversarial AutoEncoder paper)
        dcm_input_i = torch.cat(
            (latents_i, labels_i.float().view(-1, 1).expand(-1, 1)), dim=1
        )
        dcm_input_j = torch.cat(
            (latents_j, labels_j.float().view(-1, 1).expand(-1, 1)), dim=1
        )

    else:
        dcm_input_i = latents_i
        dcm_input_j = latents_j

    dcm_output_i = latent_dcm(dcm_input_i)
    dcm_output_j = latent_dcm(dcm_input_j)

    domain_labels_i = torch.zeros(dcm_output_i.size(0)).long()
    domain_labels_j = torch.ones(dcm_output_j.size(0)).long()

    dcm_loss = 0.5 * (latent_dcm_loss(dcm_output_i, domain_labels_i) + latent_dcm_loss(dcm_output_j, domain_labels_j))

    # Backpropagate loss and update parameters
    dcm_loss.backward()
    latent_dcm_optimizer.step()

    # Get summary statistics
    batch_size_i = inputs_i.size(0)
    batch_size_j = inputs_j.size(0)

    accuracy_i = accuracy(dcm_output_i, domain_labels_i)
    accuracy_j = accuracy(dcm_output_j, domain_labels_j)

    summary_stats = {'dcm_loss':dcm_loss, 'accuracy_i':accuracy_i, 'accuracy_j':accuracy_j}
    return summary_stats






def train_architecture(
    # dict consists of dictionary domain_name:(model:..., optimizer:..., data_loader_dict:..., data_key:..., label_key:...)
    domain_model_and_data_dict: dict[dict],
    latent_classifier: nn.Module,
    data_loaders_dict: dict,
    latent_optimizer: Optimizer,
    loss_dict: dict,
    n_epochs: int = 500,
    early_stopping: int = -1,
    device: device = None,
    output_dir: str = "../../data/exps/",
) -> Tuple[nn.Module, nn.Module, dict]:
    if early_stopping < 0:
        early_stopping = n_epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not device:
        device = get_device()

    start_time = time.time()

    vae = vae.to(device)
    latent_classifier = latent_classifier.to(device)

    best_vae_weights = copy.deepcopy(vae.state_dict())
    best_latent_clf_weights = copy.deepcopy(latent_classifier.state_dict())

    loss_history = {"train": [], "val": []}

    best_loss = np.infty
    early_stopping_counter = 0

    for epoch in range(n_epochs):
        logging.debug("Started epoch {}/{}".format(epoch + 1, n_epochs))
        logging.debug("--" * 20)

        if early_stopping_counter > early_stopping:
            logging.debug(
                "Training was stopped early due to no improvement of the validation"
                " loss for {} epochs.".format(early_stopping)
            )
            break

        for phase in ["train", "val"]:
            if phase == "train":
                vae.train()
                latent_classifier.train()
            else:
                vae.eval()
                latent_classifier.eval()

            running_recon_loss = 0.0
            running_regularizer_loss = 0.0
            running_latent_clf_loss = 0.0
            running_total_loss = 0.0

            for index, data in enumerate(data_loaders_dict[phase]):
                inputs = data[data_key].to(device)
                labels = data[label_key].to(device)

                # reset optimizers
                vae_optimizer.zero_grad()
                latent_optimizer.zero_grad()

                # compute loss
                with torch.set_grad_enabled(
                    phase == "train"
                ) and torch.autograd.set_detect_anomaly(False):
                    recon, z, mu, logvar = vae(inputs)
                    latent_clf_output = latent_classifier(z)

                    recon_loss = loss_dict["vae"](recon, inputs)
                    regularizer_loss = loss_dict["regularizer"](mu, logvar)
                    latent_clf_loss = loss_dict["latent_clf"](latent_clf_output, labels)

                    loss = (
                        recon_loss
                        + loss_dict["regularizer_weight"] * regularizer_loss
                        + loss_dict["latent_clf_weight"] * latent_clf_loss
                    )

                # backpropagate loss
                if phase == "train":
                    with torch.autograd.set_detect_anomaly(False):
                        loss.backward()
                        vae_optimizer.step()
                        latent_clf_loss.backward()
                        latent_optimizer.step()

                # compute batch statistics
                running_recon_loss += recon_loss.item()
                running_regularizer_loss += regularizer_loss.item()
                running_latent_clf_loss += latent_clf_loss.item()
                running_total_loss += loss.item()

            # compute epoch statistics
            n_batches = len(data_loaders_dict[phase])
            epoch_recon_loss = running_recon_loss / n_batches
            epoch_regularizer_loss = running_regularizer_loss / n_batches
            epoch_latent_clf_loss = running_latent_clf_loss / n_batches
            epoch_total_loss = running_total_loss / n_batches

            loss_history[phase].append(epoch_total_loss)

            logging.debug("{} LOSS STATISTICS FOR EPOCH {}: ".format(phase, epoch + 1))
            logging.debug("Reconstruction Loss: {:.6f}".format(epoch_recon_loss))
            logging.debug("Regularizer loss: {:.6f}".format(epoch_regularizer_loss))
            logging.debug(
                "Latent classifier loss: {:.6f}".format(epoch_latent_clf_loss)
            )
            logging.debug("***" * 20)
            logging.debug("Total loss: {:.6f}".format(epoch_total_loss))

            if phase == "val" and epoch_total_loss < best_loss:
                best_loss = epoch_total_loss
                best_vae_weights = copy.deepcopy(vae.state_dict())
                best_latent_clf_weights = copy.deepcopy(latent_classifier.state_dict())
                torch.save(vae, output_dir + "/vae.pth")
                torch.save(latent_classifier, output_dir + "/latent_clf.pth")
            elif phase == "val" and epoch_total_loss > best_loss:
                early_stopping_counter += 1

    # Training finished
    time_elapsed = time.time() - start_time
    logging.debug(
        "Training finished after {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Load best models
    vae.load_state_dict(best_vae_weights)
    latent_classifier.load_state_dict(best_latent_clf_weights)

    if "test" in data_loaders_dict:
        running_recon_loss = 0.0
        running_regularizer_loss = 0.0
        running_latent_clf_loss = 0.0
        running_total_loss = 0.0
        for index, data in enumerate(data_loaders_dict["test"]):
            inputs = data[data_key].to(device)
            labels = data[label_key].to(device)

            # compute loss
            recon, z, mu, logvar = vae(inputs)
            latent_clf_output = latent_classifier(z)

            recon_loss = loss_dict["vae"](recon, inputs)
            regularizer_loss = loss_dict["regularizer"](mu, logvar)
            latent_clf_loss = loss_dict["latent_clf"](latent_clf_output, labels)

            loss = (
                recon_loss
                + loss_dict["regularizer_weight"] * regularizer_loss
                + loss_dict["latent_clf_weight"] * latent_clf_loss
            )

            # compute batch statistics
            running_recon_loss += recon_loss.item()
            running_regularizer_loss += regularizer_loss.item()
            running_latent_clf_loss += latent_clf_loss.item()
            running_total_loss += loss.item()

        # compute epoch statistics
        n_batches = len(data_loaders_dict["test"])
        epoch_recon_loss = running_recon_loss / n_batches
        epoch_regularizer_loss = running_regularizer_loss / n_batches
        epoch_latent_clf_loss = running_latent_clf_loss / n_batches
        epoch_total_loss = running_total_loss / n_batches

        loss_history["test"] = epoch_total_loss

        logging.debug("{} LOSS STATISTICS: ".format("test"))
        logging.debug("Reconstruction Loss: {:.6f}".format(epoch_recon_loss))
        logging.debug("Regularizer loss: {:.6f}".format(epoch_regularizer_loss))
        logging.debug("Latent classifier loss: {:.6f}".format(epoch_latent_clf_loss))
        logging.debug("***" * 20)
        logging.debug("Total loss: {:.6f}".format(epoch_total_loss))

    return vae, latent_classifier, loss_history


def train_combined_architecture(
    model: nn.Module,
    data_loaders_dict: dict,
    optimizer: Optimizer,
    loss_dict: dict,
    data_key: str = "image",
    label_key: str = "label",
    n_epochs: int = 500,
    early_stopping: int = -1,
    device: device = None,
    output_dir: str = "../../data/exps/",
) -> Tuple[nn.Module, dict]:
    if early_stopping < 0:
        early_stopping = n_epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not device:
        device = get_device()

    start_time = time.time()

    model = model.to(device)

    best_model_weights = copy.deepcopy(model.state_dict())

    loss_history = {"train": [], "val": []}

    best_loss = np.infty
    early_stopping_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", verbose=True
    )

    for epoch in range(n_epochs):
        logging.debug("Started epoch {}/{}".format(epoch + 1, n_epochs))
        logging.debug("---" * 20)

        if early_stopping_counter > early_stopping:
            logging.debug(
                "Training was stopped early due to no improvement of the validation"
                " loss for {} epochs.".format(early_stopping)
            )
            break

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_recon_loss = 0.0
            running_regularizer_loss = 0.0
            running_latent_clf_loss = 0.0
            running_total_loss = 0.0

            for index, data in enumerate(data_loaders_dict[phase]):
                inputs = data[data_key].to(device)
                batch_size = inputs.size()[0]
                labels = data[label_key].to(device).view(-1)

                # reset optimizers
                optimizer.zero_grad()

                # compute loss
                with torch.set_grad_enabled(
                    phase == "train"
                ) and torch.autograd.set_detect_anomaly(False):
                    recon, z, mu, logvar, latent_clf_preds = model(inputs)
                    recon_loss = loss_dict["recon_loss"](recon, inputs)
                    regularizer_loss = loss_dict["regularizer_loss"](mu, logvar)
                    latent_clf_loss = loss_dict["latent_clf_loss"](
                        latent_clf_preds, labels
                    )

                    loss = (
                        recon_loss
                        + loss_dict["regularizer_weight"] * regularizer_loss
                        + loss_dict["latent_clf_weight"] * latent_clf_loss
                    )

                # backpropagate loss
                if phase == "train":
                    with torch.autograd.set_detect_anomaly(False):
                        loss.backward()
                        optimizer.step()

                # compute batch statistics
                running_recon_loss += recon_loss.item()
                running_regularizer_loss += regularizer_loss.item()
                running_latent_clf_loss += latent_clf_loss.item()
                running_total_loss += loss.item()

            # compute epoch statistics
            n_batches = len(data_loaders_dict[phase])
            epoch_recon_loss = running_recon_loss / n_batches
            epoch_regularizer_loss = running_regularizer_loss / n_batches
            epoch_latent_clf_loss = running_latent_clf_loss / n_batches
            epoch_total_loss = running_total_loss / n_batches

            loss_history[phase].append(epoch_total_loss)

            logging.debug(
                "{} LOSS STATISTICS FOR EPOCH {}: ".format(phase.upper(), epoch + 1)
            )
            logging.debug(
                "{} Reconstruction Loss: {:.6f}".format(phase.upper(), epoch_recon_loss)
            )
            logging.debug(
                "{} Regularizer loss: {:.6f}".format(
                    phase.upper(), epoch_regularizer_loss
                )
            )
            logging.debug(
                "{} Latent classifier loss: {:.6f}".format(
                    phase.upper(), epoch_latent_clf_loss
                )
            )
            logging.debug("***" * 20)
            logging.debug(
                "{} Total loss: {:.6f}".format(phase.upper(), epoch_total_loss)
            )
            logging.debug(" ")

            if phase == "val":
                scheduler.step(epoch_total_loss)
            if phase == "val" and epoch_total_loss < best_loss:
                best_loss = epoch_total_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model, output_dir + "/augmented_ae.pth")
                early_stopping_counter = 0
            elif phase == "val" and epoch_total_loss > best_loss:
                early_stopping_counter += 1
        logging.debug("===" * 20)
        logging.debug(" ")

    # Training finished
    time_elapsed = time.time() - start_time
    logging.debug(
        "Training finished after {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Load best models
    model.load_state_dict(best_model_weights)

    if "test" in data_loaders_dict:
        running_recon_loss = 0.0
        running_regularizer_loss = 0.0
        running_latent_clf_loss = 0.0
        running_total_loss = 0.0
        for index, data in enumerate(data_loaders_dict["test"]):
            inputs = data[data_key].to(device)
            labels = data[label_key].to(device).view(-1)

            # compute loss
            recon, z, mu, logvar, latent_clf_preds = model(inputs)

            recon_loss = loss_dict["recon_loss"](recon, inputs)
            regularizer_loss = loss_dict["regularizer_loss"](mu, logvar)
            latent_clf_loss = loss_dict["latent_clf_loss"](latent_clf_preds, labels)

            loss = (
                recon_loss
                + loss_dict["regularizer_weight"] * regularizer_loss
                + loss_dict["latent_clf_weight"] * latent_clf_loss
            )

            # compute batch statistics
            running_recon_loss += recon_loss.item()
            running_regularizer_loss += regularizer_loss.item()
            running_latent_clf_loss += latent_clf_loss.item()
            running_total_loss += loss.item()

        # compute epoch statistics
        n_batches = len(data_loaders_dict["test"])
        epoch_recon_loss = running_recon_loss / n_batches
        epoch_regularizer_loss = running_regularizer_loss / n_batches
        epoch_latent_clf_loss = running_latent_clf_loss / n_batches
        epoch_total_loss = running_total_loss / n_batches

        loss_history["test"] = epoch_total_loss

        logging.debug("{} LOSS STATISTICS: ".format("TEST"))
        logging.debug("{} Reconstruction Loss: {:.6f}".format("TEST", epoch_recon_loss))
        logging.debug(
            "{} Regularizer loss: {:.6f}".format("TEST", epoch_regularizer_loss)
        )
        logging.debug(
            "{} Latent classifier loss: {:.6f}".format("TEST", epoch_latent_clf_loss)
        )
        logging.debug("***" * 20)
        logging.debug("{} Total loss: {:.6f}".format("TEST", epoch_total_loss))
        logging.debug(" ")
        logging.debug(" ")

    return model, loss_history


def train_combined_architecture_notebook(
    model: nn.Module,
    data_loaders_dict: dict,
    optimizer: Optimizer,
    loss_dict: dict,
    data_key: str = "image",
    label_key: str = "label",
    n_epochs: int = 500,
    early_stopping: int = -1,
    visualize_performance=100,
    device: device = None,
    output_dir: str = "../../data/exps/",
) -> Tuple[nn.Module, dict]:
    if early_stopping < 0:
        early_stopping = n_epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not device:
        device = get_device()

    start_time = time.time()

    model = model.to(device)

    best_model_weights = copy.deepcopy(model.state_dict())

    loss_history = {"train": [], "val": []}

    best_loss = np.infty
    early_stopping_counter = 0

    for epoch in tnrange(n_epochs, desc="Epoch progress"):
        logging.debug("Started epoch {}/{}".format(epoch + 1, n_epochs))
        logging.debug("--" * 20)

        if early_stopping_counter > early_stopping:
            logging.debug(
                "Training was stopped early due to no improvement of the validation"
                " loss for {} epochs.".format(early_stopping)
            )
            break

        for phase in ["train", "val"]:
            logging.debug("Start phase {}".format(phase))
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_recon_loss = 0.0
            running_regularizer_loss = 0.0
            running_latent_clf_loss = 0.0
            running_total_loss = 0.0

            for j in tqdm_notebook(
                range(len(data_loaders_dict[phase])), desc="Within-epoch progress"
            ):
                data = data_loaders_dict[phase][j]
                inputs = data[data_key].to(device)
                labels = data[label_key].to(device)

                # reset optimizers
                optimizer.zero_grad()

                # compute loss
                with torch.set_grad_enabled(
                    phase == "train"
                ) and torch.autograd.set_detect_anomaly(False):
                    recons, z, mu, logvar, clf_preds = model(inputs)

                    recon_loss = loss_dict["vae"](recons, inputs)
                    regularizer_loss = loss_dict["regularizer"](mu, logvar)
                    latent_clf_loss = loss_dict["latent_clf"](clf_preds, labels)

                    loss = (
                        recon_loss
                        + loss_dict["regularizer_weight"] * regularizer_loss
                        + loss_dict["latent_clf_weight"] * latent_clf_loss
                    )

                # backpropagate loss
                if phase == "train":
                    with torch.autograd.set_detect_anomaly(False):
                        loss.backward()
                        optimizer.step()

                # compute batch statistics
                running_recon_loss += recon_loss.item()
                running_regularizer_loss += regularizer_loss.item()
                running_latent_clf_loss += latent_clf_loss.item()
                running_total_loss += loss.item()

            # Visualize performance
            if epoch % visualize_performance == 0 and phase == "val":
                latent_representations = model.reparameterize(mu, logvar)
                samples = model.decode(latent_representations)
                input_fig, output_fig, sample_fig = visualize_model_performance(
                    inputs=inputs,
                    outputs=recons,
                    samples=samples,
                    labels=labels,
                    label_dict=None,
                )
                input_fig.show()
                output_fig.show()
                sample_fig.show()

            # compute epoch statistics
            n_batches = len(data_loaders_dict[phase])
            epoch_recon_loss = running_recon_loss / n_batches
            epoch_regularizer_loss = running_regularizer_loss / n_batches
            epoch_latent_clf_loss = running_latent_clf_loss / n_batches
            epoch_total_loss = running_total_loss / n_batches

            loss_history[phase].append(epoch_total_loss)

            logging.debug("{} LOSS STATISTICS FOR EPOCH {}: ".format(phase, epoch + 1))
            logging.debug("Reconstruction Loss: {:.6f}".format(epoch_recon_loss))
            logging.debug("Regularizer loss: {:.6f}".format(epoch_regularizer_loss))
            logging.debug(
                "Latent classifier loss: {:.6f}".format(epoch_latent_clf_loss)
            )
            logging.debug("***" * 20)
            logging.debug("Total loss: {:.6f}".format(epoch_total_loss))

            if phase == "val" and epoch_total_loss < best_loss:
                best_loss = epoch_total_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model, output_dir + "/augmented_ae.pth")
            elif phase == "val" and epoch_total_loss > best_loss:
                early_stopping_counter += 1

    # Training finished
    time_elapsed = time.time() - start_time
    logging.debug(
        "Training finished after {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Load best models
    model.load_state_dict(best_model_weights)

    if "test" in data_loaders_dict:
        running_recon_loss = 0.0
        running_regularizer_loss = 0.0
        running_latent_clf_loss = 0.0
        running_total_loss = 0.0
        for index, data in enumerate(data_loaders_dict["test"]):
            inputs = data[data_key].to(device)
            labels = data[label_key].to(device)

            # compute loss
            recon, z, mu, logvar, clf_preds = model(inputs)

            recon_loss = loss_dict["vae"](recon, inputs)
            regularizer_loss = loss_dict["regularizer"](mu, logvar)
            latent_clf_loss = loss_dict["latent_clf"](clf_preds, labels)

            loss = (
                recon_loss
                + loss_dict["regularizer_weight"] * regularizer_loss
                + loss_dict["latent_clf_weight"] * latent_clf_loss
            )

            # compute batch statistics
            running_recon_loss += recon_loss.item()
            running_regularizer_loss += regularizer_loss.item()
            running_latent_clf_loss += latent_clf_loss.item()
            running_total_loss += loss.item()

        # compute epoch statistics
        n_batches = len(data_loaders_dict["test"])
        epoch_recon_loss = running_recon_loss / n_batches
        epoch_regularizer_loss = running_regularizer_loss / n_batches
        epoch_latent_clf_loss = running_latent_clf_loss / n_batches
        epoch_total_loss = running_total_loss / n_batches

        loss_history["test"] = epoch_total_loss

        logging.debug("{} LOSS STATISTICS: ".format("test"))
        logging.debug("Reconstruction Loss: {:.6f}".format(epoch_recon_loss))
        logging.debug("Regularizer loss: {:.6f}".format(epoch_regularizer_loss))
        logging.debug("Latent classifier loss: {:.6f}".format(epoch_latent_clf_loss))
        logging.debug("***" * 20)
        logging.debug("Total loss: {:.6f}".format(epoch_total_loss))

    return model, loss_history


def get_loss_dict(
    vae_loss: str = "mse",
    latent_clf_loss: str = "ce",
    latent_clf_class_weights: List = None,
    regularizer_loss: str = "kld",
    latent_clf_weight: float = 1.0,
    regularizer_weight: float = 1.0,
) -> dict:
    device = get_device()

    if vae_loss == "l2":
        vae_loss = nn.MSELoss()
    elif vae_loss == "l1":
        vae_loss = nn.L1Loss()
    elif vae_loss == "bce":
        vae_loss = nn.BCELoss()
    else:
        raise NotImplementedError("Unknown loss type: {}".format(vae_loss))

    if latent_clf_loss == "ce":
        latent_clf_loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(latent_clf_class_weights).to(device)
        )
    elif latent_clf_loss == "bce":
        latent_clf_loss = nn.BCELoss(
            weight=torch.FloatTensor(latent_clf_class_weights).to(device)
        )
    else:
        raise NotImplementedError("Unknown loss type: {}".format(vae_loss))

    if regularizer_loss == "kld":
        regularizer_loss = KLDLoss()
    else:
        raise NotImplementedError

    loss_dict = {
        "recon_loss": vae_loss,
        "latent_clf_loss": latent_clf_loss,
        "regularizer_loss": regularizer_loss,
        "latent_clf_weight": latent_clf_weight,
        "regularizer_weight": regularizer_weight,
    }
    return loss_dict
