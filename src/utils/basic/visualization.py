from typing import List, Tuple

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, Axes
from torch import Tensor
import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader
from src.models.vae import BaseVAE
from src.utils.torch.general import get_device


def visualize_vae_performance(
    vae_model: BaseVAE,
    data_loader: DataLoader,
    data_key: str = "images",
    label_key: str = "label",
    label_dict: dict = None,
    n_samples:int = 16
) -> Tuple[Figure, Figure, Figure]:
    device = get_device()
    vae_model.to(device)
    data = next(iter(data_loader))
    inputs = data[data_key].to(device)
    labels = data[label_key].to(device)
    outputs, z, mu, logvar, _ = vae_model(inputs)
    samples = vae_model.sample(len(inputs), device=device)
    # samples=None

    input_fig, output_fig, sample_fig = visualize_model_performance(
        inputs=inputs[:n_samples],
        outputs=outputs[:n_samples],
        labels=labels[:n_samples],
        samples=samples[:n_samples],
        label_dict=label_dict,
    )
    return input_fig, output_fig, sample_fig


def visualize_model_performance(
    inputs: Tensor,
    outputs: Tensor,
    labels: Tensor,
    samples: Tensor = None,
    label_dict: dict = None,
) -> Tuple[Figure, Figure, Figure]:
    batch_size = inputs.size()[0]
    grid_width = int(np.rint(np.sqrt(batch_size)))
    grid_height = int(np.rint(batch_size / grid_width))

    # Get labels
    labels = labels.view(-1).cpu().numpy().astype(str)
    if label_dict:
        labels = np.array([label_dict[l] for l in labels])

    input_images = inputs.clone().detach().cpu().numpy()
    input_fig, input_axes = get_image_grid_plot(
        images=input_images,
        grid_width=grid_width,
        grid_height=grid_height,
        labels=labels,
    )
    input_fig.suptitle("Input images")

    output_images = outputs.clone().detach().cpu().numpy()
    output_fig, output_axes = get_image_grid_plot(
        images=output_images,
        grid_width=grid_width,
        grid_height=grid_height,
        labels=labels,
    )
    output_fig.suptitle("Reconstructed images")

    sample_fig = None

    if samples is not None:
        sample_images = samples.clone().detach().cpu().numpy()
        sample_fig, sample_axes = get_image_grid_plot(
            images=sample_images, grid_width=grid_width, grid_height=grid_height
        )
        sample_fig.suptitle("Sampled images")

    return input_fig, output_fig, sample_fig


def get_image_grid_plot(
    images: ndarray, grid_width: int, grid_height: int, labels: List = None
) -> Tuple[Figure, Axes]:
    n_images = len(images)
    fig, axes = plt.subplots(ncols=grid_width, nrows=grid_height, figsize=[20, 20])
    for i in range(grid_width):
        for j in range(grid_height):
            if i * grid_height + j > n_images - 1:
                break
            else:
                image = images[i * grid_height + j][0]
                image = np.rint(image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                axes[i, j].imshow(image, cmap="gray")
                if labels is not None and len(labels) > i * grid_height + j:
                    axes[i, j].set_title("Type {}".format(labels[i * grid_height + j]))
                    axes[i, j].axis("off")
    return fig, axes
