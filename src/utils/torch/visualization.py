import os
from typing import List

import cv2
import imageio
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from src.helper.models import DomainModelConfig
from src.utils.basic.visualization import deprocess_image, show_cam_on_image


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

    image_model = domain_model_configs[0].model.to(device).eval()
    rna_model = domain_model_configs[1].model.to(device).eval()

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


def visualize_geneset_perturbation_in_image(data_dict:dict, output_dir:str, silencing_node:int):
    image_dir = os.path.join(output_dir, "perturbation/silenced_set_{}".format(silencing_node))
    os.makedirs(image_dir)
    trans_images = data_dict['trans_images']
    perturbed_trans_images = data_dict['perturbed_trans_images']

    for i in range(len(trans_images)):
        imageio.imwrite(os.path.join(image_dir, "trans_image_%s.jpg" % i) ,
                                   np.uint8(trans_images[i].squeeze() * 255))
        imageio.imwrite(os.path.join(image_dir, "perturbed_trans_image_%s.jpg" % i) ,
                                   np.uint8(perturbed_trans_images[i].squeeze() * 255))
        plt.figure()
        plt.imshow(trans_images[i].squeeze() * 255, cmap='gray')
        plt.imshow((perturbed_trans_images[i].squeeze() - trans_images[i].squeeze()) * 255, cmap='seismic', alpha=0.3)
        plt.axis("off")
        plt.savefig(os.path.join(image_dir, "perturbed_diff_image_%s.jpg" % i), bbox_inches='tight')
        plt.close()


def visualize_geneset_guided_grad_cams(data_dict:dict, output_dir:str, query_node:int):
    grad_cams = data_dict['grad_cams']
    gb_maps = data_dict['gb_maps']
    images = data_dict['images']
    image_dir = os.path.join(output_dir, "guided_gradcam/queried_set_{}".format(query_node))
    os.makedirs(image_dir)

    for i in range(len(grad_cams)):
        grayscale_cam = grad_cams[i].squeeze()
        image = images[i].squeeze()
        cam = show_cam_on_image(image, grayscale_cam)
        gb = gb_maps[i]
        gb = cv2.cvtColor(gb, cv2.COLOR_GRAY2RGB)
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        cv2.imwrite(os.path.join(image_dir, "gradcam_image_%s.jpg" % i), cam)
        cv2.imwrite(os.path.join(image_dir, "gb_map_image_%s.jpg" % i), gb)
        cv2.imwrite(os.path.join(image_dir, "guided_gradcam_image_%s.jpg" % i), cam_gb)