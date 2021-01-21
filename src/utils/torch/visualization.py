import os
from typing import List

import imageio
import numpy as np
import torch
from torch.autograd import Variable

from src.helper.models import DomainModelConfig


def visualize_image_translation_performance(
        domain_model_configs: List[DomainModelConfig],
        epoch: int,
        output_dir: str,
        phase:str,
        device: str = "cuda:0",
):
    # Todo make more generic
    image_dir = os.path.join(output_dir, "epoch_{}/images/{}".format(epoch, phase))
    os.makedirs(image_dir, exist_ok=True)

    image_model = domain_model_configs[0].model.to(device)
    rna_model = domain_model_configs[1].model.to(device)

    rna_inputs = domain_model_configs[1].inputs
    rna_inputs = Variable(rna_inputs).to(device)

    image_inputs = domain_model_configs[0].inputs
    image_inputs = Variable(image_inputs).to(device)

    rna_latents = rna_model(rna_inputs)["latents"]
    translated_images = image_model.decode(rna_latents)
    for i in range(min(image_inputs.size()[0], rna_inputs.size()[0])):
        imageio.imwrite(
            os.path.join(image_dir, "epoch_%s_inputs_%s.jpg" % (epoch, i)),
            np.uint8(image_inputs[i].squeeze().cpu().data.numpy() * 255),
        )
        imageio.imwrite(
            os.path.join(image_dir, "epoch_%s_trans_%s.jpg" % (epoch, i)),
            np.uint8(translated_images[i].squeeze().cpu().data.numpy() * 255),
        )
        recon_images = image_model(image_inputs)['recons']
        imageio.imwrite(
            os.path.join(image_dir, "epoch_%s_recon_%s.jpg" % (epoch, i)),
            np.uint8(recon_images[i].squeeze().cpu().data.numpy() * 255),
        )


def visualize_image_vae_performance(
        domain_model_config: DomainModelConfig,
        epoch: int,
        output_dir: str,
        phase: str,
        device: str = "cuda:0",
):
    image_dir = os.path.join(output_dir, "epoch_{}/images".format(epoch))
    os.makedirs(image_dir, exist_ok=True)

    image_vae = domain_model_config.model.to(device)
    image_inputs = domain_model_config.inputs.to(device)

    recon_images = image_vae(image_inputs)["recons"]
    sample_images = image_vae.sample(5, device=device)

    for i in range(10):
        imageio.imwrite(
            os.path.join(image_dir, "%s_epoch_%s_inputs_%s.jpg" % (phase,epoch, i)),
            np.uint8(image_inputs[i].cpu().data.view(64, 64).numpy() * 255),
        )
        imageio.imwrite(
            os.path.join(image_dir, "%s_epoch_%s_recons_%s.jpg" % (phase,epoch, i)),
            np.uint8(recon_images[i].cpu().data.view(64, 64).numpy() * 255),
        )
        imageio.imwrite(
            os.path.join(image_dir, "%s_epoch_%s_samples_%s.jpg" % (phase,epoch, i)),
            np.uint8(sample_images[i].cpu().data.view(64, 64).numpy() * 255),
        )


def visualize_image_ae_performance(
        domain_model_config: DomainModelConfig,
        epoch: int,
        output_dir: str,
        phase: str,
        device: str = "cuda:0",
):
    image_dir = os.path.join(output_dir, "epoch_{}/images".format(epoch))
    os.makedirs(image_dir, exist_ok=True)

    image_ae = domain_model_config.model.to(device)
    image_inputs = domain_model_config.inputs.to(device)

    recon_images = image_ae(image_inputs)["recons"]

    for i in range(image_inputs.size()[0]):
        imageio.imwrite(
            os.path.join(image_dir, "%s_epoch_%s_inputs_%s.jpg" % (phase,epoch, i)),
            np.uint8(image_inputs[i].cpu().data.view(64, 64).numpy() * 255),
        )
        imageio.imwrite(
            os.path.join(image_dir, "%s_epoch_%s_recons_%s.jpg" % (phase,epoch, i)),
            np.uint8(recon_images[i].cpu().data.view(64, 64).numpy() * 255),
        )
