import numpy as np
import matplotlib.pyplot as plt
import os


def draw_his(version, show=False):
    loss = np.load(os.path.join("./logs", "loss_his_{}.npy".format(version)))
    acc = np.load(os.path.join("./logs", "acc_his_{}.npy".format(version)))
    loss_val = np.load(os.path.join("./logs", "loss_val_his_{}.npy".format(version)))
    acc_val = np.load(os.path.join("./logs", "acc_val_his_{}.npy".format(version)))
    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.plot(loss)
    plt.subplot(1, 2, 2)
    plt.plot(loss_val)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join("./logs", "loss val_{}.png".format(version)))
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(acc)
    plt.subplot(1, 2, 2)
    plt.plot(acc_val)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join("./logs", "acc val_{}.png".format(version)))


if __name__ == "__main__":
    draw_his("resnet-2.0", True)
