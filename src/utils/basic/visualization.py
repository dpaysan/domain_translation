import logging
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.helper.models import DomainConfig
from src.utils.torch.evaluation import (
    get_shared_latent_space_dict_for_multiple_domains,
    get_full_latent_space_dict_for_multiple_domains,
    save_latents_to_csv,
)

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_train_val_hist(
    training_history: ndarray,
    validation_history: ndarray,
    output_dir: str,
    y_label: str,
    title=None,
    posfix: str = "",
):
    r""" A function to visualize the evolution of the training and validation loss during the training.
    Parameters
    ----------
    training_history : list, numpy.ndarray
        The training loss for the individual training epochs.
    validation_history : list, numpy.ndarray
        The validation lss for the individual training epochs.
    output_dir : str
        The path of the directory the visualization of the evolution of the loss is stored in.
    y_label : str
        The label of the y-axis of the visualization.
    title : None, str
        The title of the visualization. If ``None`` is given, it is `'Fitting History` by default.
    posfix : str
        An additional posfix for the file path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    epochs = np.arange(len(training_history))
    lines = plt.plot(epochs, training_history, epochs, validation_history)
    plt.ylabel(y_label)
    plt.xlabel("Epoch")
    plt.legend(
        ("Training Loss", "Validation Loss", "Validation loss"),
        loc="upper right",
        markerscale=2.0,
    )
    if title is None:
        title = "Fitting History"
    plt.title(title)
    plt.savefig(output_dir + "plotted_fitting_hist{}.png".format(posfix))
    plt.close()


def plot_latent_representations(
    latents_domain_dict: dict,
    save_path: str,
    random_state: int,
    reduction: str = "umap",
    label_dict: dict = None,
):
    latent_representations = []
    domain_names = []
    labels = []

    for domain_name, latents in latents_domain_dict.items():
        domain_names.append([domain_name] * len(latents))
        latent_representations.append(latents)
        if label_dict is not None and domain_name in label_dict:
            labels.append(label_dict[domain_name])

    latent_representations = np.concatenate(latent_representations)
    domain_names = np.concatenate(domain_names)

    if len(labels) > 0:
        labels = np.concatenate(labels)
    else:
        labels = None

    # Scale the data (sub-optimal should use same scaling for different datasets)
    latent_representations = StandardScaler().fit_transform(latent_representations)

    if reduction == "umap":
        mapper = UMAP(random_state=random_state)
        transformed = mapper.fit_transform(latent_representations)
    elif reduction == "umap_labeled":
        mapper = UMAP(random_state=random_state)
        transformed = mapper.fit_transform(latent_representations, labels)
    elif reduction == "tsne":
        mapper = TSNE(random_state=random_state)
        transformed = mapper.fit_transform(latent_representations)
    elif reduction == "pca":
        mapper = PCA(n_components=2, random_state=random_state)
        transformed = mapper.fit_transform(latent_representations)
    else:
        raise RuntimeError("Unknown reduction mode encountered {}".format(reduction))

    transformed = pd.DataFrame(
        data=transformed, columns=[reduction + "-c1", reduction + "-c2"]
    )

    # Plot the data

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=reduction + "-c1",
        y=reduction + "-c2",
        hue=domain_names,
        palette=sns.color_palette("dark", len(set(domain_names))),
        data=transformed,
        legend="full",
        alpha=0.4,
        s=60,
    )

    plt.legend(markerscale=2.0)

    if labels is not None:
        label_point(
            transformed.iloc[:, 0],
            transformed.iloc[:, 1],
            pd.Series(labels.astype(np.int), dtype=int),
            plt.gca(),
        )

    plt.savefig(save_path)
    plt.close()


def label_point(x, y, val, ax):
    a = pd.concat({"x": x, "y": y, "val": val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point["x"] + 0.02, point["y"], str(int(point["val"])), fontsize=12)


def visualize_shared_latent_space(
    domain_configs: List[DomainConfig],
    save_path: str,
    random_state: int,
    reduction: str = "umap",
    dataset_type: str = "val",
    device: str = "cuda:0",
):
    latents_domain_dict, label_dict = get_shared_latent_space_dict_for_multiple_domains(
        domain_configs=domain_configs, dataset_type=dataset_type, device=device
    )
    plot_latent_representations(
        latents_domain_dict=latents_domain_dict,
        save_path=save_path,
        random_state=random_state,
        reduction=reduction,
        label_dict=label_dict,
    )


def plot_correlations_latent_space(latents_domain_dict, save_path):
    shared_latents_in_domains = []
    domain_specific_latents_in_domains = []
    shared_colnames = []
    specific_colnames = []
    for domain_name, latents in latents_domain_dict.items():
        shared_latents, domain_specific_latents = latents
        shared_latents_in_domains.append(shared_latents)

        shared_colnames.append(
            np.array(
                [
                    "z_{}_{}".format(domain_name, i)
                    for i in range(shared_latents.shape[1])
                ]
            )
        )
        if domain_specific_latents is not None:
            domain_specific_latents_in_domains.append(domain_specific_latents)
            specific_colnames.append(
                np.array(
                    [
                        "n_{}_{}".format(domain_name, i)
                        for i in range(domain_specific_latents.shape[1])
                    ]
                )
            )

    if len(domain_specific_latents_in_domains) == 0:
        latent_reps = np.concatenate(shared_latents_in_domains, axis=1)
        colnames = np.concatenate(shared_colnames)
    else:
        latent_reps = np.concatenate(
            [
                np.concatenate(shared_latents_in_domains, axis=1),
                np.concatenate(domain_specific_latents_in_domains, axis=1),
            ],
            axis=1,
        )
        colnames = np.concatenate(
            [np.concatenate(shared_colnames), np.concatenate(specific_colnames)]
        )

    latents = pd.DataFrame(data=latent_reps, columns=list(colnames))

    # Plot latent space correlation
    plt.figure(figsize=(20, 15))
    ax = plt.axes()
    sns.heatmap(latents.corr(), ax=ax, cmap="coolwarm")
    plt.savefig(save_path)
    plt.close()


def visualize_correlation_structure_latent_space(
    domain_configs: List[DomainConfig],
    save_path: str,
    dataset_type: str = "val",
    device: str = "cuda:0",
):
    latents_domain_dict, _ = get_full_latent_space_dict_for_multiple_domains(
        domain_configs=domain_configs, dataset_type=dataset_type, device=device
    )
    plot_correlations_latent_space(
        latents_domain_dict=latents_domain_dict, save_path=save_path
    )


def visualize_model_performance(
    output_dir: str,
    domain_configs: List[DomainConfig],
    dataset_types: List[str] = None,
    device: str = "cuda:0",
):
    os.makedirs(output_dir, exist_ok=True)
    if dataset_types is None:
        dataset_types = ["train", "val"]

    for dataset_type in dataset_types:
        visualize_shared_latent_space(
            domain_configs=domain_configs,
            save_path=output_dir
            + "/shared_latent_space_umap_{}.png".format(dataset_type),
            dataset_type=dataset_type,
            random_state=42,
            reduction="umap",
            device=device,
        )

        visualize_correlation_structure_latent_space(
            domain_configs=domain_configs,
            save_path=output_dir
            + "/latent_space_correlation_structure_{}.png".format(dataset_type),
            dataset_type=dataset_type,
            device=device,
        )

    for domain_config in domain_configs:
        domain_name = domain_config.name
        for dataset_type in dataset_types:
            save_latents_to_csv(
                domain_config=domain_config,
                save_path=output_dir
                + "/{}_latent_representations_{}.csv".format(domain_name, dataset_type),
                dataset_type=dataset_type,
                device=device,
            )


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
