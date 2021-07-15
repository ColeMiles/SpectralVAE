import matplotlib.pyplot as plt
import numpy as np


def plot_training(train_losses, val_losses, learning_rates):
    fig, loss_ax = plt.subplots()
    lr_ax = plt.twinx(loss_ax)

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    if len(train_losses.shape) < 2:
        train_losses = np.expand_dims(train_losses, axis=1)
        val_losses = np.expand_dims(val_losses, axis=1)

    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    lr_ax.set_ylabel("Learning Rate")

    loss_ax.plot(train_losses[:, 0], 'k', lw=2, label="Train Reconstruct")
    loss_ax.plot(val_losses[:, 0], 'r', lw=2, label="Val Reconstruct")

    if train_losses.shape[1] == 2:
        loss_ax.plot(train_losses[:, 1], 'b', lw=2, label="Train KL Div")

    loss_ax.set_yscale('log')
    lr_ax.plot(learning_rates, '--', color='purple', label="Learning Rate")

    fig.legend()
    fig.tight_layout()

    return fig, (lr_ax, loss_ax)
