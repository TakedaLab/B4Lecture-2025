"""Solution to ex4 on Gaussian mixture models."""

import matplotlib.pyplot as plt
import numpy as np


def em_clustering(data, n_clusters):
    """Perform GMM soft clustering by the EM algorithm.

    Params:
        data (ndarray): The data to cluster.
        n_clusters (int): The number of desired clusters.
    Returns:
    """
    # Turn 1-D data vector into matrix if needed
    if len(data.shape) == 1:
        data = data[:, np.newaxis]

    # Initialize cluster parameters [mixture probability, mean, variance]
    gmm = [
        {
            "p": 1.0 / n_clusters,
            "mean": np.random.rand(data.shape[1]),
            "cov": np.identity(data.shape[1]),
        }
        for _ in range(n_clusters)
    ]
    return gmm


def plot_scatter(data: np.ndarray, title: str, filename: str, mm=None) -> None:
    """Create and save scatterplot as image file.

    Can optionally take a mixture model as a parameter to plot on top of the data points.

    Params:
        data (ndarray): The input data to be plotted.
        mm (): The optional mixture model to plot on top of the scatterplot.
        title: The desired plot title.
        filename: Path to the desired output file.
    Returns: None
    """
    if len(data.shape) == 1:
        # 1-D data (added violin plot for better visualisation)
        violin = plt.violinplot(
            data, orientation="horizontal", showextrema=True, showmeans=True
        )["bodies"][0]
        violin.set_facecolor("#bf5cb2")
        plt.scatter(data, np.ones_like(data), c="#912583", zorder=2)
        plt.xlim(-3, 3)
        plt.title(title)
        plt.savefig(filename)
        plt.clf()
    else:
        plt.scatter(data[:, 0], data[:, 1], s=15, c="#912583", zorder=2)
        plt.title(title)
        plt.grid()
        plt.savefig(filename)
        plt.clf()


def main() -> None:
    """Execute main routine.

    Params: None
    Returns: None
    """
    # Data loading
    data1 = np.loadtxt("ex4/data1.csv", delimiter=",")
    data2 = np.loadtxt("ex4/data2.csv", delimiter=",")
    data3 = np.loadtxt("ex4/data3.csv", delimiter=",")

    # Scatterplots

    # Data 1 (added violin plot for better visualisation of 1-D data)
    plot_scatter(
        data1, title="Data 1 Scatterplot", filename="ex4/i_tzimas/data1_scatter.png"
    )

    # Data 2
    plot_scatter(
        data2, title="Data 2 Scatterplot", filename="ex4/i_tzimas/data2_scatter.png"
    )

    # Data 3
    plot_scatter(
        data3, title="Data 3 Scatterplot", filename="ex4/i_tzimas/data3_scatter.png"
    )

    # Clustering
    # print(em_clustering(data1, 3))


if __name__ == "__main__":
    main()
