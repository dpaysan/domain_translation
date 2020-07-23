import copy
from typing import Tuple, List

from torch import nn, device
import os
from src.utils.torch.general import get_device
import time
import numpy as np
import logging
import torch
from torch.optim.optimizer import Optimizer
from tqdm import tnrange, tqdm_notebook
from src.utils.basic.visualization import visualize_model_performance
from src.functions.loss_functions import KLDLoss, BCELoss_transformed


def train_architecture(
    vae: nn.Module,
    latent_classifier: nn.Module,
    data_loaders_dict: dict,
    vae_optimizer: Optimizer,
    latent_optimizer: Optimizer,
    loss_dict: dict,
    data_key: str = "image",
    label_key: str = "label",
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
                early_stopping_counter=0
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
