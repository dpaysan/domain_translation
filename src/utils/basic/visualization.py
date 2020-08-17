from numpy import ndarray
import os
import matplotlib.pyplot as plt
import logging
import numpy as np


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
