import torch
from torch.nn import (
    Module,
    # CrossEntropyLoss,
    L1Loss,
    MSELoss,
    BCELoss,
    BCEWithLogitsLoss,
)

from src.helper.custom_losses import (
    RobustMSELoss,
    RobustL1Loss,
    RobustBCELoss,
    RobustCrossEntropyLoss,
    RobustBCEWithLogitsLoss,
)
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.rmsprop import RMSprop

from src.helper.models import DomainConfig
from src.models.ae import VanillaAE, TwoLatentSpaceAE, VanillaConvAE, GeneSetAE
from src.models.custom_networks import GeneSetAE_v2
from src.models.latent_models import (
    LatentDiscriminator,
    LatentClassifier,
    LatentRegressor,
)
from src.models.gmvae import GaussianMixtureVAE, GaussianMixtureConvVAE
from src.models.vae import VanillaConvVAE, VanillaVAE, GeneSetVAE
from src.utils.torch.general import get_device


def add_model_to_optimizer(optimizer: Optimizer, model: Module):
    optimizer.add_param_group(model.parameters())


def get_optimizer_for_model(optimizer_dict: dict, model: Module) -> Optimizer:
    optimizer_type = optimizer_dict.pop("type")
    if optimizer_type == "adam":
        optimizer = Adam(model.parameters(), **optimizer_dict)
    elif optimizer_type == "rmsprop":
        optimizer = RMSprop(model.parameters(), **optimizer_dict)
    else:
        raise NotImplementedError('Unknown optimizer type "{}"'.format(optimizer_type))
    return optimizer


def get_domain_configuration(
    name: str,
    model_dict: dict,
    optimizer_dict: dict,
    recon_loss_fct_dict: dict,
    data_loader_dict: dict,
    data_key: str,
    label_key: str,
    train_model: bool = True,
) -> DomainConfig:

    model_type = model_dict.pop("type")
    if model_type == "VanillaConvVAE":
        model = VanillaConvVAE(**model_dict)
    elif model_type == "VanillaConvAE":
        model = VanillaConvAE(**model_dict)
    elif model_type == "VanillaVAE":
        model = VanillaVAE(**model_dict)
    elif model_type == "VanillaAE":
        model = VanillaAE(**model_dict)
    elif model_type == "TwoLatentSpaceAE":
        model = TwoLatentSpaceAE(**model_dict)
    elif model_type == "GaussianMixtureVAE":
        model = GaussianMixtureVAE(**model_dict)
    elif model_type == "GaussianMixtureConvVAE":
        model = GaussianMixtureConvVAE(**model_dict)
    elif model_type == "GeneSetAE":
        model = GeneSetAE(**model_dict)
    elif model_type == "GeneSetAE_v2":
        model = GeneSetAE_v2(**model_dict)
    elif model_type == "GeneSetVAE":
        model = GeneSetVAE(**model_dict)
    else:
        raise NotImplementedError('Unknown model type "{}"'.format(model_type))

    optimizer = get_optimizer_for_model(optimizer_dict=optimizer_dict, model=model)

    recon_loss_fct_type = recon_loss_fct_dict.pop("type")
    if recon_loss_fct_type == "mae":
        recon_loss_function = L1Loss()
    elif recon_loss_fct_type == "mse":
        recon_loss_function = MSELoss()
    elif recon_loss_fct_type == "bce":
        recon_loss_function = BCELoss()
    elif recon_loss_fct_type == "bce_ll":
        recon_loss_function = BCEWithLogitsLoss()
    else:
        raise NotImplementedError(
            'Unknown loss function type "{}"'.format(recon_loss_fct_type)
        )

    domain_config = DomainConfig(
        name=name,
        model=model,
        optimizer=optimizer,
        recon_loss_function=recon_loss_function,
        data_loader_dict=data_loader_dict,
        data_key=data_key,
        label_key=label_key,
        train_model=train_model,
    )

    return domain_config


def get_latent_model_configuration(
    model_dict: dict, optimizer_dict: dict, loss_dict: dict, device: None
) -> dict:

    if device is None:
        device = get_device()

    model_type = model_dict.pop("type")
    if model_type == "LatentDiscriminator":
        model = LatentDiscriminator(**model_dict)
    elif model_type == "LatentClassifier":
        model = LatentClassifier(**model_dict)
    elif model_type == "LatentRegressor":
        model = LatentRegressor(**model_dict)
    else:
        raise NotImplementedError('Unknown model type "{}"'.format(model_type))

    optimizer = get_optimizer_for_model(optimizer_dict=optimizer_dict, model=model)

    if model_type != "LatentRegressor":
        try:
            weights = torch.FloatTensor(loss_dict.pop("weights")).to(device)
        except KeyError:
            weights = torch.ones(model_dict["n_classes"]).float().to(device)
    else:
        weights = None

    loss_type = loss_dict.pop("type")
    if loss_type == "ce":
        latent_loss = RobustCrossEntropyLoss(weight=weights)
    elif loss_type == "bce":
        latent_loss = RobustBCELoss(weight=weights)
    elif loss_type == "bce_ll":
        latent_loss = RobustBCEWithLogitsLoss(weight=weights)
    elif loss_type == "mse":
        latent_loss = RobustMSELoss()
    elif loss_type == "mae":
        latent_loss = RobustL1Loss()
    else:
        raise NotImplementedError('Unknown loss type "{}"'.format(loss_type))

    latent_model_config = {"model": model, "optimizer": optimizer, "loss": latent_loss}
    return latent_model_config
