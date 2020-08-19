from typing import List

from numpy import ndarray
import os
import matplotlib.pyplot as plt
import logging
import numpy as np
from umap import UMAP
from umap.plot import points
from sklearn.manifold import TSNE
from pandas import DataFrame
import seaborn as sns
import pandas as pd

from src.helper.models import DomainConfig
from src.utils.torch.evaluation import get_latent_space_dict_for_multiple_domains


def plot_train_val_hist(
    training_history: ndarray,
    validation_history: ndarray,
    output_dir: str,
    y_label: str,
    title=None,
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
        ("Training Loss", "Validation Loss", "Validation loss"), loc="upper right"
    )
    if title is None:
        title = "Fitting History"
    plt.title(title)
    plt.savefig(output_dir + "plotted_fitting_hist.png")
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

    if reduction == "umap":
        mapper = UMAP(random_state=random_state)
        transformed = mapper.fit_transform(latent_representations)
    elif reduction == "umap_labeled":
        mapper = UMAP(random_state=random_state)
        transformed = mapper.fit_transform(latent_representations, labels)
    elif reduction == "tsne":
        mapper = TSNE(random_state=random_state)
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
        alpha=0.7,
    )
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
        ax.text(point["x"] + 0.02, point["y"], str(int(point["val"])), fontsize=6)


def visualize_shared_latent_space(
    domain_configs: List[DomainConfig],
    save_path: str,
    random_state: int,
    reduction: str = "umap",
    dataset_type: str = "val",
    device: str = "cuda:0",
):
    latents_domain_dict, label_dict = get_latent_space_dict_for_multiple_domains(
        domain_configs=domain_configs, dataset_type=dataset_type, device=device
    )
    plot_latent_representations(
        latents_domain_dict=latents_domain_dict,
        save_path=save_path,
        random_state=random_state,
        reduction=reduction,
        label_dict=label_dict,
    )