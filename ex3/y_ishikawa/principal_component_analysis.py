"""Apply principal component analysis to data and visualize."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIMENSION_2D = 2
DIMENSION_3D = 3


class PCA:
    """A class to perform Principal Component Analysis (PCA) on input data.

    Attributes
    ----------
    data : np.ndarray
        Input data array where each row is a sample and each column is a feature.
    transform_matrix : np.ndarray
        A matrix whose columns are the principal component vectors (eigenvectors).
    eig_val : np.ndarray
        Array of eigenvalues corresponding to the principal components.
    contributions : np.ndarray
        Contribution ratio of each principal component.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the PCA object.

        Parameters
        ----------
        df : pd.DataFrame
            Input data where each row is a sample and each column is a feature.
        """
        self.data = df.to_numpy()
        self.transform_matrix, self.eig_val = self._calc_transform_matrix()
        self.contributions = self._calc_contribution()

    def _calc_transform_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the eigenvectors and eigenvalues from the covariance matrix.

        Returns
        -------
        eig_vec : np.ndarray
            Matrix whose columns are eigenvectors sorted in descending order of eigenvalues.
        eig_val : np.ndarray
            Array of eigenvalues sorted in descending order.
        """
        # calculate covariance matrix
        cov_mat = np.cov(self.data.T)

        # calculate eigenvalues and eigenvectors
        eig_val, eig_vec = np.linalg.eig(cov_mat)

        # sort based on eigenvalues
        idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]

        return eig_vec, eig_val

    def _calc_contribution(self) -> np.ndarray:
        """Calculate the contribution ratio for each principal component.

        Returns
        -------
        contributions : np.ndarray
            Contribution ratio of each principal component.
        """
        lambda_sum = np.sum(self.eig_val)
        return self.eig_val / lambda_sum

    def dimension_based_reduction(
        self, new_dimension: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce the data to a specified number of dimensions.

        Parameters
        ----------
        new_dimension : int
            Number of principal components to retain.

        Returns
        -------
        reduced_data : np.ndarray
            Dimension-reduced data of shape (n_samples, new_dimension).
        transform_mat : np.ndarray
            Transformation matrix used for dimensionality reduction.
        """
        transform_mat = self.transform_matrix[:, :new_dimension]
        reduced_data = self.data @ transform_mat
        return reduced_data, transform_mat

    def contribution_based_reduction(
        self, contribution: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce the data to the smallest number of dimensions.

        Parameters
        ----------
        contribution : float
            Minimum cumulative contribution ratio to retain (e.g., 0.9 for 90%).

        Returns
        -------
        reduced_data : np.ndarray
            Dimension-reduced data.
        transform_mat : np.ndarray
            Transformation matrix used for dimensionality reduction.
        """
        tmp_sum = 0
        for i, rate in enumerate(self.contributions, start=1):
            tmp_sum += rate
            if tmp_sum >= contribution:
                transform_mat = self.transform_matrix[:, :i]
                reduced_data = self.data @ transform_mat
                return reduced_data, transform_mat


def plot_scatter(  # noqa: PLR0913
    df: pd.DataFrame,
    pca: PCA,
    output_dir: Path,
    title: str = "Scatter plot",
    xlabel: str = r"$x_1$",
    ylabel: str = r"$x_2$",
    zlabel: str = r"$x_3$",
) -> None:
    """Show a scatter plot from a DataFrame and saves it to `output_dir`.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing the data (2D or 3D) to be plotted.

    pca: PCA
        The PCA class.

    output_dir : Path
        The path where the plot will be saved.

    title : str, optional, default "Scatter plot"
        The title of the scatter plot.

    xlabel : str, optional, default r"$x_1$"
        The label for the x-axis.

    ylabel : str, optional, default r"$x_2$"
        The label for the y-axis.

    zlabel : str, optional, default r"$x_3$"
        The label for the z-axis, used only for 3D plots.
    """
    # reset plt
    fig = plt.figure()

    # 2-dimensions
    if df.shape[1] == DIMENSION_2D:
        ax = fig.add_subplot(111)
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], marker=".")

        # plot principal component axes
        center = (np.median(df.iloc[:, 0]), np.median(df.iloc[:, 1]))
        ax.axline(
            center,
            slope=pca.transform_matrix[0][1] / pca.transform_matrix[0][0],
            label=f"Contribution rate: {pca.contributions[0]:.3f}",
            color="r",
        )
        ax.axline(
            center,
            slope=pca.transform_matrix[1][1] / pca.transform_matrix[1][0],
            label=f"Contribution rate: {pca.contributions[1]:.3f}",
            color="g",
        )
    # 3-dimensions
    elif df.shape[1] == DIMENSION_3D:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], marker=".")

        ax.set_zlim(min(df.iloc[:, 2]) * 1.1, max(df.iloc[:, 2]) * 1.1)
        ax.set_zlabel(zlabel)

        # plot principal component axes
        center = np.array(
            [
                np.median(df.iloc[:, 0]),
                np.median(df.iloc[:, 1]),
                np.median(df.iloc[:, 2]),
            ]
        )
        colors = ["r", "g", "b"]
        t = np.array([-50, 50])
        for i, (eig_vec, contribution) in enumerate(
            zip(pca.transform_matrix.T, pca.contributions, strict=True)
        ):
            line = t[:, np.newaxis] * eig_vec + center
            ax.plot(
                line[:, 0],
                line[:, 1],
                line[:, 2],
                label=f"Contribution rate: {contribution:3.3f}",
                color=colors[i],
            )

    # set labels
    ax.set_title(title)
    ax.set_xlim(min(df.iloc[:, 0]) * 1.1, max(df.iloc[:, 0]) * 1.1)
    ax.set_ylim(min(df.iloc[:, 1]) * 1.1, max(df.iloc[:, 1]) * 1.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    # save figure
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir)

    # show graph
    plt.show()


class NameSpace:
    """Configuration namespace for PCA parameters.

    Parameters
    ----------
    output_dir : Path
        Path where the processed output will be saved.
    """

    output_dir: Path


def parse_args() -> NameSpace:
    """Parse command-line arguments.

    Returns
    -------
    arguments : NameSpace
        Parsed arguments including output directory.
    """
    # prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output result directory.",
    )

    return parser.parse_args(namespace=NameSpace())


if __name__ == "__main__":
    # get arguments
    args = parse_args()
    output_dir = args.output_dir

    # load data
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_DIR = (SCRIPT_DIR / "../data/").resolve()
    DATA1_PATH = DATA_DIR / "data1.csv"
    DATA2_PATH = DATA_DIR / "data2.csv"
    DATA3_PATH = DATA_DIR / "data3.csv"
    DATA4_PATH = DATA_DIR / "data4.csv"
    df1 = pd.read_csv(DATA1_PATH)
    df2 = pd.read_csv(DATA2_PATH)
    df3 = pd.read_csv(DATA3_PATH)
    df4 = pd.read_csv(DATA4_PATH)
    df1.attrs["title"] = DATA1_PATH.stem
    df2.attrs["title"] = DATA2_PATH.stem
    df3.attrs["title"] = DATA3_PATH.stem
    df4.attrs["title"] = DATA4_PATH.stem

    # process PCA
    pca1 = PCA(df1)
    pca2 = PCA(df2)
    pca3 = PCA(df3)
    pca4 = PCA(df4)

    # plot principal component axes
    for df, pca in zip([df1, df2, df3], [pca1, pca2, pca3], strict=True):
        title = df.attrs["title"]
        plot_scatter(
            df,
            pca,
            output_dir / "principal_component_axes" / f"{title}.png",
            title=title,
        )

    # plot PCA result
    reduced_data, _ = pca3.dimension_based_reduction(2)
    plt.figure()
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.title("Dimension reduced data3")
    plt.xlabel("new x-axis")
    plt.ylabel("new y-axis")
    plt.savefig(output_dir / "data3_PCA_result.png")
    plt.show()

    # find minimum dimension exceed 0.9 cumulative contribution rate
    _, transform_mat = pca4.contribution_based_reduction(0.9)
    print(
        f"Dimensions needed for ≥0.9 contribution: {transform_mat.shape[1]}-dimension"
    )
    print(
        f"Cumulative contribution: {sum(pca4.contributions[: transform_mat.shape[1]])}"
    )
