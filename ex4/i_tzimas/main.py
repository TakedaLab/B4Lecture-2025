"""Solution to ex4 on Gaussian mixture models."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def em_clustering(data, n_clusters, n_epochs=100) -> tuple[list, np.ndarray]:
    """Perform GMM soft clustering by the EM algorithm.

    Params:
        data (ndarray): The data to cluster.
        n_clusters (int): The number of desired clusters.
        n_epochs (int): The number of iterations to run clustering for. Default is 100.
    Returns:
        gmm, likelihoods (list, ndarray): The output GMM as a list of dictionaries, each
        representing a cluster and the array of size n_epochs storing the log likelihood
        of the model for each epoch.
    """
    # Turn 1-D data vector into matrix if needed
    if len(data.shape) == 1:
        data = data[:, np.newaxis]

    # Data sample size
    data_len = data.shape[0]

    # Initialize cluster parameters [mixture probability, mean, variance]
    gmm = [
        {
            "p": 1.0 / n_clusters,
            "mean": np.random.rand(data.shape[1]),
            "cov": np.identity(data.shape[1]),
        }
        for _ in range(n_clusters)
    ]

    # Init log likelihood array
    likelihoods = np.zeros(n_epochs)

    # Init probability matrix used in fitting
    x_probs = np.zeros((data_len, n_clusters))

    # EM loop
    for i in range(n_epochs):
        # Expectation step

        # Calculate probability of each data point to belong in each cluster
        for j in range(n_clusters):
            x_probs[:, j] = gmm[j]["p"] * multivariate_normal.pdf(
                data, gmm[j]["mean"], gmm[j]["cov"]
            )
        x_probs_sums = np.sum(x_probs, 1)[:, np.newaxis]
        x_probs /= x_probs_sums

        # Maximization step
        for j in range(n_clusters):
            x_probs_j = x_probs[:, j][:, np.newaxis]
            prob_sum = np.sum(x_probs[:, j])

            mean_j = np.sum(x_probs_j * data, 0) / prob_sum

            gmm[j]["p"] = prob_sum / data_len
            gmm[j]["mean"] = mean_j
            gmm[j]["cov"] = (x_probs_j * (data - mean_j)).T @ (data - mean_j) / prob_sum

        # Log likelihood
        likelihoods[i] = np.sum(np.log(x_probs_sums))

    return gmm, likelihoods


def plot_scatter(data: np.ndarray, title: str, filename: str, mm: list = None) -> None:
    """Create and save scatterplot as image file.

    Can optionally take a mixture model as a parameter to plot on top of the data points.

    Params:
        data (ndarray): The input data to be plotted.
        title (str): The desired plot title.
        filename (str): Path to the desired output file.
        mm (): The optional mixture model to plot on top of the scatterplot.
    Returns: None
    """
    if len(data.shape) == 1:
        if mm:
            # Plot clusters from mixture model
            x = np.linspace(np.min(data), np.max(data), 200)
            pdf = np.zeros(len(x))

            for j, cluster in enumerate(mm):
                # Calculate the GMM PDF
                pdf += (
                    multivariate_normal(cluster["mean"], cluster["cov"]).pdf(x)
                    * cluster["p"]
                )
                # Centroid
                plt.scatter(
                    cluster["mean"][0],
                    0,
                    zorder=3,
                    marker="+",
                    c="yellow",
                    s=60,
                    label="Centroids" if j == 0 else "_nolegend_",
                )

            plt.plot(x, pdf, label="GMM")
            plt.scatter(data, np.zeros_like(data), c="#912583", zorder=2)
            plt.legend()
        else:
            # 1-D data scatterplot (added violin plot for better visualisation)
            violin = plt.violinplot(
                data,
                orientation="horizontal",
                showextrema=True,
                showmeans=True,
            )["bodies"][0]
            violin.set_facecolor("#bf5cb2")
            plt.scatter(data, np.ones_like(data), c="#912583", zorder=2)

        plt.xlim(-3, 3)
        plt.title(title)
        plt.savefig(filename)
        plt.clf()
    else:
        if mm:
            # Plot clusters from mixture model
            cmaps = ["Blues", "Greens", "Oranges", "YlOrBr"]

            # Meshgrid
            x, y = np.meshgrid(
                np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 200),
                np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 200),
            )
            xy = np.stack([x.ravel(), y.ravel()]).T

            # Contour labels
            artists = []

            for j, cluster in enumerate(mm):
                # Plot a contour for each cluster
                pdf = (
                    multivariate_normal(cluster["mean"], cluster["cov"]).pdf(xy)
                    * cluster["p"]
                )
                contour = plt.contour(x, y, np.reshape(pdf, x.shape), cmap=cmaps[j])
                # Contour label
                artist, _ = contour.legend_elements()
                artists.append(artist[5])

                # Centroid
                centroid = plt.scatter(
                    cluster["mean"][0],
                    cluster["mean"][1],
                    zorder=3,
                    marker="+",
                    c="yellow",
                    s=150,
                )
                # Centroid label
                if j == 0:
                    centroid.set_label("Centroids")
                    artists = [centroid] + artists
            labels = ["Centroids"] + [f"GMM Cluster {i}" for i, _ in enumerate(mm)]

            plt.legend(artists, labels)

        # 2-D Data scatterplot
        plt.scatter(data[:, 0], data[:, 1], s=15, c="#912583", zorder=2)
        plt.title(title)
        plt.grid()
        plt.savefig(filename)
        plt.clf()


def plot_likelihood(likelihoods: np.ndarray, title: str, filename: str) -> None:
    """Plot the log likelihood as training progresses.

    Params:
        likelihoods (ndarray): Array of log likelihood for each epoch.
        title (str): The desired plot title.
        filename (str): Path to the desired output file.
    Returns: None
    """
    plt.plot(np.arange(1, len(likelihoods) + 1), likelihoods)
    plt.title(title)
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
    gmm1, lh1 = em_clustering(data1, 2)
    gmm2, lh2 = em_clustering(data2, 3)
    gmm3, lh3 = em_clustering(data3, 3)

    # Scatterplots with clustering results
    plot_scatter(
        data1,
        title="Data 1 Clustering",
        filename="ex4/i_tzimas/data1_clusters.png",
        mm=gmm1,
    )
    plot_scatter(
        data2,
        title="Data 2 Clustering",
        filename="ex4/i_tzimas/data2_clusters.png",
        mm=gmm2,
    )
    plot_scatter(
        data3,
        title="Data 3 Clustering",
        filename="ex4/i_tzimas/data3_clusters.png",
        mm=gmm3,
    )

    # Log likelihood plots
    plot_likelihood(
        lh1, title="Data 1 Log likelihood", filename="ex4/i_tzimas/data1_lh.png"
    )
    plot_likelihood(
        lh2, title="Data 2 Log likelihood", filename="ex4/i_tzimas/data2_lh.png"
    )
    plot_likelihood(
        lh3, title="Data 3 Log likelihood", filename="ex4/i_tzimas/data3_lh.png"
    )


if __name__ == "__main__":
    main()
