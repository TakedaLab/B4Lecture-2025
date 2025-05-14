"""Apply EM algorithm to data and visualize."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

DIMENSION_1D = 1
DIMENSION_2D = 2


def calc_gaussian(
    data: np.ndarray, centroid: np.ndarray, covariance_matrix: np.ndarray
) -> np.ndarray:
    """Calculate the multivariate Gaussian probability density for each data.

    Parameters
    ----------
    data : np.ndarray, shape=(n_samples, n_features)
        A 2D array representing the data points.

    centroid : np.ndarray, shape=(n_features,)
        A 1D array representing the mean (centroid) of the Gaussian distribution.

    covariance_matrix : np.ndarray, shape=(n_features, n_features)
        A 2D array representing the covariance matrix of the Gaussian distribution.

    Returns
    -------
    probability_density: np.ndarray, shape=(n_samples,)
        A 1D array containing the probability density for each data point.

    """
    data_num, data_dim = data.shape
    result = np.zeros(data_num)
    for n, datum in enumerate(data):
        diff = datum - centroid
        result[n] = np.exp(
            -diff.T @ np.linalg.inv(covariance_matrix) @ diff / 2
        ) / np.sqrt((2 * np.pi) ** data_dim * np.linalg.det(covariance_matrix))

    return result


def calc_mix_gaussian(
    data: np.ndarray,
    weights: np.ndarray,
    centroids: np.ndarray,
    covariance_matrices: np.ndarray,
) -> np.ndarray:
    """Calculate the probability density of a mixture of Gaussian distributions for each data.

    Parameters
    ----------
    data : np.ndarray, shape=(n_samples, n_features)
        A 2D array representing the data points.

    weights : np.ndarray, shape=(n_clusters,)
        A 1D array representing the mixture weights for each Gaussian component.

    centroids : np.ndarray, shape=(n_clusters, n_features)
        A 2D array representing the centroids (means) of each Gaussian component.

    covariance_matrices : np.ndarray, shape=(n_clusters, n_features, n_features)
        A 3D array representing the covariance matrices of each Gaussian component.

    Returns
    -------
    probability_density: np.ndarray, shape=(n_samples,)
        A 1D array containing the probability density for each data point.

    """
    data_num, _ = data.shape
    result = np.zeros(data_num)
    for weight, centroid, covariance_matrix in zip(
        weights, centroids, covariance_matrices, strict=True
    ):
        result += weight * calc_gaussian(data, centroid, covariance_matrix)

    return result


def calc_log_likelihood(
    data: np.ndarray,
    weights: np.ndarray,
    centroids: np.ndarray,
    covariance_matrices: np.ndarray,
) -> float:
    """Calculate the log-likelihood of the data under a Gaussian mixture model.

    Parameters
    ----------
    data : np.ndarray, shape=(n_samples, n_features)
        A 2D array representing the data points.

    weights : np.ndarray, shape=(n_clusters,)
        A 1D array representing the mixture weights for each Gaussian component.

    centroids : np.ndarray, shape=(n_clusters, n_features)
        A 2D array representing the centroids (means) of each Gaussian component.

    covariance_matrices : np.ndarray, shape=(n_clusters, n_features, n_features)
        A 3D array representing the covariance matrices of each Gaussian component.

    Returns
    -------
    log_likelihood : float
        The log-likelihood of the data under the Gaussian mixture model.

    """
    mix_gaussian = calc_mix_gaussian(data, weights, centroids, covariance_matrices)
    log_likelihood = np.sum(np.log(mix_gaussian))

    return log_likelihood


class EMAlgorithm:
    """A class to perform EM Algorithm on input data for Gaussian Mixture Model (GMM).

    Attributes
    ----------
    input_data : np.ndarray
        Input data array where each row is a sample and each column is a feature.
    cluster_num : int
        The number of Gaussian clusters to be used in the GMM.
    data_num : int
        The number of data points in the input data.
    data_dim : int
        The number of features of the data.
    weights : np.ndarray
        Array of weights for each Gaussian component in the mixture model.
    centroids : np.ndarray
        Array of centroids for each Gaussian component in the mixture model.
    covariance_matrices : np.ndarray
        Array of covariance matrices for each Gaussian component in the mixture model.
    likelihood_history : list
        A list of log-likelihood values for each iteration of the EM algorithm.

    """

    def __init__(self, input_data: pd.DataFrame, cluster_num: int) -> None:
        """Initialize the EM algorithm object.

        Parameters
        ----------
        input_data : pd.DataFrame
            Input data where each row is a sample and each column is a feature.
        cluster_num : int
            The number of Gaussian clusters to be used in the GMM.

        """
        self.input_data = input_data.to_numpy()
        self.cluster_num = cluster_num
        self.data_num, self.data_dim = self.input_data.shape

        # initialize GMM parameters
        self.weights, self.centroids, self.covariance_matrices = (
            self.get_initial_gmm_parameters()
        )

        self.likelihood_history = []

    def get_initial_gmm_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize the parameters for the Gaussian Mixture Model (GMM).

        Returns
        -------
        weights : np.ndarray, shape=(n_clusters,)
            Array of initial weights for each Gaussian component in the mixture model.
        centroids : np.ndarray, shape=(n_clusters, n_features)
            Array of initial centroids (mean vectors) for each Gaussian component.
        covariance_matrices : np.ndarray, shape=(n_clusters, n_features, n_features)
            Array of initial covariance matrices for each Gaussian component.

        """
        weights = np.ones(self.cluster_num) / self.cluster_num
        centroids = np.random.randn(self.cluster_num, self.data_dim)
        covariance_matrices = np.array(
            [np.eye(self.data_dim) for _ in range(self.cluster_num)]
        )

        return weights, centroids, covariance_matrices

    def em_algorithm(self, threshold: float = 1e-6) -> None:
        """Run the EM algorithm until convergence.

        Parameters
        ----------
        threshold : float, optional
            Convergence threshold for log-likelihood. Default is 1e-6.

        """
        pre_weights, pre_centroids, pre_covariance_matrices = (
            self.get_initial_gmm_parameters()
        )
        pre_likelihood = calc_log_likelihood(
            self.input_data, pre_weights, pre_centroids, pre_covariance_matrices
        )
        likelihood = calc_log_likelihood(
            self.input_data, self.weights, self.centroids, self.covariance_matrices
        )
        self.likelihood_history.append(likelihood)
        while abs(likelihood - pre_likelihood) > threshold:
            pre_weights = self.weights
            pre_centroids = self.centroids
            pre_covariance_matrices = self.covariance_matrices
            pre_likelihood = likelihood

            responsibility = self.e_step()

            self.weights, self.centroids, self.covariance_matrices = self.m_step(
                responsibility
            )
            likelihood = calc_log_likelihood(
                self.input_data, self.weights, self.centroids, self.covariance_matrices
            )
            self.likelihood_history.append(likelihood)

    def e_step(self) -> np.ndarray:
        """Perform the E-step of the EM algorithm.

        Returns
        -------
        responsibility : np.ndarray, shape=(n_samples, n_clusters)
            Matrix of responsibilities.

        """
        responsibility = np.zeros((self.data_num, self.cluster_num))
        mix_gaussian = calc_mix_gaussian(
            self.input_data, self.weights, self.centroids, self.covariance_matrices
        )
        for k, (weight, centroid, covariance_matrix) in enumerate(
            zip(self.weights, self.centroids, self.covariance_matrices, strict=True)
        ):
            weighted_gaussian = weight * calc_gaussian(
                self.input_data, centroid, covariance_matrix
            )
            responsibility[:, k] = weighted_gaussian / mix_gaussian

        return responsibility

    def m_step(
        self, responsibility: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform the M-step of the EM algorithm.

        Parameters
        ----------
        responsibility : np.ndarray, shape=(n_samples, n_clusters)
            Matrix of responsibilities.

        Returns
        -------
        new_weights : np.ndarray, shape=(n_clusters,)
            Updated weights for each Gaussian component.
        new_centroids : np.ndarray, shape=(n_clusters, n_features)
            Updated centroids (mean vectors) for each Gaussian component.
        new_covariance_matrices : np.ndarray, shape=(n_clusters, n_features, n_features)
            Updated covariance matrices for each Gaussian component.

        """
        N_k = np.sum(responsibility, axis=0)

        new_weights = N_k / self.data_num
        new_centroids = (responsibility.T @ self.input_data) / N_k[:, np.newaxis]
        new_covariance_matrices = np.zeros(
            (self.cluster_num, self.data_dim, self.data_dim)
        )
        for k in range(self.cluster_num):
            diff = self.input_data - self.centroids[k]
            new_covariance_matrices[k] = (responsibility[:, k] * diff.T @ diff) / N_k[k]

        return new_weights, new_centroids, new_covariance_matrices

    def cluster_analyses(self) -> np.ndarray:
        """Assign each data point to the cluster with the highest responsibility.

        Returns
        -------
        labels : np.ndarray, shape=(n_samples,)
            Array of cluster indices assigned to each data point.

        """
        responsibility = self.e_step()
        return np.argmax(responsibility, axis=1)

    def plot(self, output_path: Path, title: str = "Scatter plot") -> None:
        """Visualize the results of the EM algorithm including the likelihood history.

        Parameters
        ----------
        output_path : Path
            Path to save the generated plot.
        title : str, optional
            Title for the clustering result plot. Default is "Scatter plot".

        """
        _, axs = plt.subplots(1, 2, figsize=(10, 4))

        # plot likelihood history
        axs[0].plot(range(len(self.likelihood_history)), self.likelihood_history)
        axs[0].set_title("Likelihood history")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Likelihood")
        if self.data_dim == DIMENSION_1D:
            # plot data
            cmap = cm.get_cmap("Set3", self.cluster_num)
            colors = [cmap(label) for label in self.cluster_analyses()]
            plt.scatter(
                self.input_data,
                np.zeros(self.data_num),
                edgecolors=colors,
                facecolors="none",
                label="Data sample",
            )

            # plot centroid
            plt.scatter(
                self.centroids,
                np.zeros(self.cluster_num),
                marker="x",
                color="red",
                label="Centroids",
            )

            # plot probability density
            x = np.linspace(
                min(self.input_data) * 1.1, max(self.input_data) * 1.1, 10**3
            )
            plt.plot(
                x,
                calc_mix_gaussian(
                    np.reshape(x, (len(x), 1)),
                    self.weights,
                    self.centroids,
                    self.covariance_matrices,
                ),
                label="GMM",
            )

            axs[1].set_title(f"{title} (K = {self.cluster_num})")
            axs[1].set_xlabel(r"$x$")
            axs[1].set_ylabel("Probability density")
        elif self.data_dim == DIMENSION_2D:
            # plot data
            cmap = cm.get_cmap("Set3", self.cluster_num)
            colors = [cmap(label) for label in self.cluster_analyses()]
            plt.scatter(
                self.input_data[:, 0],
                self.input_data[:, 1],
                edgecolors=colors,
                facecolors="none",
                label="Data sample",
            )

            # plot centroid
            plt.scatter(
                self.centroids[:, 0],
                self.centroids[:, 1],
                marker="x",
                color="red",
                label="Centroids",
            )

            # plot probability density
            x = np.linspace(
                min(self.input_data[:, 0]) * 1.1,
                max(self.input_data[:, 0]) * 1.1,
                200,
            )
            y = np.linspace(
                min(self.input_data[:, 1]) * 1.1,
                max(self.input_data[:, 1]) * 1.1,
                200,
            )
            X, Y = np.meshgrid(x, y)
            Z = calc_mix_gaussian(
                np.column_stack([X.ravel(), Y.ravel()]),
                self.weights,
                self.centroids,
                self.covariance_matrices,
            ).reshape(X.shape)
            contour = axs[1].contour(X, Y, Z)
            plt.colorbar(contour)

            axs[1].set_title(f"{title} (K = {self.cluster_num})")
            axs[1].set_xlabel(r"$x$")
            axs[1].set_ylabel(r"$y$")
        plt.legend()
        plt.savefig(output_path)


class NameSpace:
    """Configuration namespace for linear regression processing parameters.

    Parameters
    ----------
    input_paths: list[Path]
        List of paths to input data.
    output_dir : Path
        Path where the processed output will be saved.
    cluster_nums : list[int]
        List of cluster numbers to be used in EM algorithm.

    """

    input_paths: list[Path]
    output_dir: Path
    cluster_nums: list[int]


def parse_args() -> NameSpace:
    """Parse command-line arguments.

    Returns
    -------
    arguments : NameSpace
        Parsed arguments including input/output paths, cluster numbers parameters.

    """
    # data path
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_DIR = (SCRIPT_DIR / "../").resolve()
    DATA1_PATH = DATA_DIR / "data1.csv"
    DATA2_PATH = DATA_DIR / "data2.csv"
    DATA3_PATH = DATA_DIR / "data3.csv"

    # prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output result directory.",
    )
    parser.add_argument(
        "--input_paths",
        type=Path,
        nargs="*",
        default=[DATA1_PATH, DATA2_PATH, DATA3_PATH],
        help="Path to input data.",
    )
    parser.add_argument(
        "--cluster_nums",
        type=int,
        nargs="*",
        default=[2, 3, 3],
        help="Degree of linear regression.",
    )

    return parser.parse_args(namespace=NameSpace())


if __name__ == "__main__":
    # get arguments
    args = parse_args()
    input_paths = args.input_paths
    output_dir = args.output_dir
    cluster_nums = args.cluster_nums

    # load data
    data: list[pd.DataFrame] = []
    for input_path in input_paths:
        df = pd.read_csv(input_path, header=None)
        df.attrs["title"] = input_path.stem
        data.append(df)

    # plot EM algorithm results
    np.random.seed(42)
    for df, cluster_num in zip(data, cluster_nums, strict=True):
        title = df.attrs["title"]
        em_class = EMAlgorithm(df, cluster_num)
        em_class.em_algorithm()
        em_class.plot(output_dir / f"{title}.png", title)
        plt.show()
