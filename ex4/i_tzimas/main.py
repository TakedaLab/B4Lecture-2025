"""Solution to ex4 on Gaussian mixture models."""

import matplotlib.pyplot as plt
import numpy as np


def main():
    """Execute main routine.

    Params:
    Returns:
    """
    # Data loading
    data1 = np.loadtxt("ex4/data1.csv", delimiter=",")
    data2 = np.loadtxt("ex4/data2.csv", delimiter=",")
    data3 = np.loadtxt("ex4/data3.csv", delimiter=",")

    # Scatterplots

    # Data 1
    violin = plt.violinplot(
        data1, orientation="horizontal", showextrema=True, showmeans=True
    )["bodies"][0]
    violin.set_facecolor("#bf5cb2")
    plt.scatter(data1, np.ones_like(data1), c="#912583", zorder=2)
    plt.xlim(-3, 3)
    plt.title("Data 1 Scatterplot")
    plt.savefig("ex4/i_tzimas/data1_scatter.png")


if __name__ == "__main__":
    main()
