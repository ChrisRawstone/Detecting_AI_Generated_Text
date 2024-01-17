import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wandb
import os


def plot_confusion_matrix_sklearn(y, y_pred, display_labels=None, run=None, save_path=None, name=None):
    """
    Plots the confusion matrix for a given sklearn model.

    Parameters:
    - model: The sklearn model for which the confusion matrix is to be plotted.
    - X: Input features for prediction.
    - y: True labels.
    - display_labels: List of labels to be displayed on the confusion matrix plot.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()

    # Save confusion matrix plot to wandb
    if run:
        run.log({"confusion_matrix": wandb.Image(plt)}, commit=False)

    if save_path:
        plt.savefig(os.path.join(save_path, name))

    plt.close()
