"""Apply linear regression to data and visualize."""

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIMENSION_2D = 2
DIMENSION_3D = 3


def plot_scatter(  # noqa: PLR0913
    df: pd.DataFrame,
    output_dir: Path,
    title: str = "Scatter plot",
    weights: np.ndarray | None = None,
    xlabel: str = r"$x_1$",
    ylabel: str = r"$x_2$",
    zlabel: str = r"$x_3$",
) -> None:
    """Show a scatter plot from a DataFrame and saves it to `output_dir`.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing the data (2D or 3D) to be plotted.

    output_dir : Path
        The path where the plot will be saved.

    title : str, optional, default "Scatter plot"
        The title of the scatter plot.

    weights: np.ndarray, optional, default None
        The weights for regression graph.

    xlabel : str, optional, default r"$x_1$"
        The label for the x-axis.

    ylabel : str, optional, default r"$x_2$"
        The label for the y-axis.

    zlabel : str, optional, default r"$x_3$"
        The label for the z-axis, used only for 3D plots.
    """

    def generate_label(weights: np.ndarray, dimension: int) -> str:
        """Generate a LaTeX-formatted regression equation label for legends.

        Parameters
        ----------
        weights : np.ndarray
            Coefficients of the regression model.
        dimension : int
            Dimensionality of the regression model.

        Returns
        -------
        label : str
            A LaTeX-formatted string representing the regression equation.
        """
        if dimension == DIMENSION_2D:
            label = rf"${ylabel.replace('$', '')} = {round(weights[0], 3)}"
            for d, weight in enumerate(weights[1:], start=1):
                label += rf"+ {round(weight, 3)}{xlabel.replace('$', '')}^{d}"
            label += r"$"
        elif dimension == DIMENSION_3D:
            degree = len(weights) // 2

            label = rf"${zlabel.replace('$', '')} = {round(weights[0], 3)}"
            for d, weight in enumerate(weights[1:], start=1):
                if d <= degree:
                    label += rf"+ {round(weight, 3)}{xlabel.replace('$', '')}^{d}"
                else:
                    label += (
                        rf"+ {round(weight, 3)}{ylabel.replace('$', '')}^{d - degree}"
                    )
            label += r"$"

        return label

    # reset plt
    fig = plt.figure()

    # flip weights
    if weights is not None:
        weights = weights[::-1]

    # 2-dimensions
    if df.shape[1] == DIMENSION_2D:
        ax = fig.add_subplot(111)
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], marker=".")

        if weights is not None:
            f = np.poly1d(weights)
            x = np.linspace(min(df.iloc[:, 0]), max(df.iloc[:, 0]), 10**3)
            y = f(x)
            ax.plot(
                x, y, label=generate_label(weights[::-1], df.shape[1]), color="orange"
            )
    # 3-dimensions
    elif df.shape[1] == DIMENSION_3D:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], marker=".")

        ax.set_zlabel(zlabel)

        if weights is not None:
            degree = len(weights) // 2

            f = np.poly1d(weights[degree:])
            g = np.poly1d(np.append(weights[:degree], 0))
            x = np.linspace(min(df.iloc[:, 0]), max(df.iloc[:, 0]), 10**3)
            y = np.linspace(min(df.iloc[:, 1]), max(df.iloc[:, 1]), 10**3)
            x, y = np.meshgrid(x, y)
            z = f(x) + g(y)
            surface = ax.plot_surface(
                x,
                y,
                z,
                label=generate_label(weights[::-1], df.shape[1]),
                color="orange",
                alpha=0.5,
            )

    # set labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if weights is not None:
        if df.shape[1] == DIMENSION_3D:
            surface._facecolors2d = surface._facecolor3d
            surface._edgecolors2d = surface._edgecolor3d
        ax.legend()

    # save figure
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir)

    # show graph
    plt.show()


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Apply soft thresholding to the input array.

    Parameters
    ----------
    x : np.ndarray
        Input array of values to be thresholded.
    lam : float
        Threshold value. Elements with absolute value less than `lam` will be zero.

    Returns
    -------
    np.ndarray
        Array after applying soft thresholding.
    """
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def linear_regression(  # noqa: PLR0913
    df: pd.DataFrame,
    degree: int,
    norm: str = "",
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    threshold: float = 1e-7,
) -> np.ndarray:
    """Perform polynomial linear regression using least squares.

    Parameters
    ----------
    df : pd.DataFrame, shape=(samples, dimension)
        Input data where the first (dimension - 1) columns are explanatory variables
        and the last column is the predictor variable.
    degree : int
        The degree of the polynomial to fit for each input feature.
    norm : Literal["ridge", "lasso", "elastic_net", ""]
        Method of normalization to apply. ("ridge", "lasso", "elastic_net")
    lambda1 : float
        Parameter used in L1 norm (lasso, elastic_net).
    lambda2 : float
        Parameter used in L2 norm (ridge, elastic_net).
    threshold:
        Threshold to stop weight update loop (lasso, elastic_net).

    Returns
    -------
    weights: np.ndarray
        Coefficients of the polynomial linear regression model.
    """
    # get size
    data_num, dimension = df.shape
    term_num = (dimension - 1) * degree + 1

    # convert data
    # 1 + x_0 + x_0^2 + ... + x_0^degree + x_1 + x_1^2 + ... + x_1^degree + ...
    X = np.ones((data_num, term_num))
    for i in range(dimension - 1):
        for d in range(1, degree + 1):
            X[:, i * degree + d] = df.iloc[:, i] ** d
    y = df.iloc[:, -1]

    # OLS
    match norm:
        # Lasso norm
        case "lasso":
            # calculate square sum
            square_sum = np.sum(X**2, axis=0)

            weights = np.random.rand(term_num)
            pre_weights = np.ones_like(weights)
            # loop until update stop
            while np.sum(np.abs(weights - pre_weights)) > threshold:
                # copy weights
                pre_weights = weights.copy()

                # update bias
                weights[0] = np.sum(y - X[:, 1:] @ weights[1:]) / data_num
                # update weights
                for i in range(1, term_num):
                    tmp = (y - np.delete(X, i, 1) @ np.delete(weights, i)) @ X[:, i]

                    weights[i] = soft_threshold(tmp, lambda1) / square_sum[i]
            return weights
        # Ridge norm
        case "ridge":
            return np.linalg.inv(X.T @ X + lambda2 * np.eye(X.shape[1])) @ X.T @ y
        # Elastic Net
        case "elastic_net":
            # calculate square sum
            square_sum = np.sum(X**2, axis=0)

            weights = np.random.rand(term_num)
            pre_weights = np.ones_like(weights)
            # loop until update stop
            while np.sum(np.abs(weights - pre_weights)) > threshold:
                # copy weights
                pre_weights = weights.copy()

                # update bias
                weights[0] = np.sum(y - X[:, 1:] @ weights[1:]) / (
                    data_num + 2 * lambda2
                )
                # update weights
                for i in range(1, term_num):
                    tmp = (y - np.delete(X, i, 1) @ np.delete(weights, i)) @ X[:, i]

                    weights[i] = soft_threshold(tmp, lambda1) / (
                        square_sum[i] + 2 * lambda2
                    )
            return weights
        case _:
            return np.linalg.inv(X.T @ X) @ X.T @ y


class NameSpace:
    """Configuration namespace for linear regression processing parameters.

    Parameters
    ----------
    input_paths: list[Path]
        List of paths to input data.
    output_dir : Path
        Path where the processed output will be saved.
    degrees : list[int]
        List of polynomial degrees to be used in regression.
    norm : Literal["lasso", "ridge", "elastic_net", ""]
        Method of normalization to apply. ("lasso", "ridge", "elastic_net")
    lambda1 : float
        Parameter used in L1 norm (lasso, elastic_net).
    lambda2 : float
        Parameter used in L2 norm (ridge, elastic_net).
    threshold : float
        Threshold to stop weight update loop (lasso, elastic_net).
    """

    input_paths: list[Path]
    output_dir: Path
    degrees: list[int]
    norm: Literal["lasso", "ridge", "elastic_net", ""]
    lambda1: float
    lambda2: float
    threshold: float


def parse_args() -> NameSpace:
    """Parse command-line arguments.

    Returns
    -------
    arguments : NameSpace
        Parsed arguments including input/output paths, regression degrees, parameters.
    """
    # data path
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_DIR = (SCRIPT_DIR / "../data/").resolve()
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
        "--degrees",
        type=int,
        nargs="*",
        default=[1, 3, 2],
        help="Degree of linear regression.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="",
        choices=["lasso", "ridge", "elastic_net"],
        help="Method of normalization. (lasso, ridge, elastic_net).",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.1,
        help="Parameter for L1 norm (lasso, elastic_net).",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.1,
        help="Parameter for L2 norm (ridge, elastic_net).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-7,
        help="Threshold to stop weight update loop (lasso, elastic_net).",
    )

    return parser.parse_args(namespace=NameSpace())


if __name__ == "__main__":
    # get arguments
    args = parse_args()
    input_paths = args.input_paths
    output_dir = args.output_dir
    degrees = args.degrees
    norm = args.norm
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    threshold = args.threshold

    # load data
    data: list[pd.DataFrame] = []
    for input_path in input_paths:
        df = pd.read_csv(input_path)
        df.attrs["title"] = input_path.stem
        data.append(df)

    # plot original data
    for df in data:
        title = df.attrs["title"]
        plot_scatter(df, output_dir / "scatter_plot" / f"{title}.png", title=title)

    # plot linear regression results
    for df, degree in zip(data, degrees, strict=True):
        # estimate weight with linear regression
        weights = linear_regression(
            df, degree, norm=norm, lambda1=lambda1, lambda2=lambda2, threshold=threshold
        )

        title = df.attrs["title"]
        plot_scatter(
            df,
            output_dir / "linear_regression" / f"{title}.png",
            title=title,
            weights=weights,
        )
