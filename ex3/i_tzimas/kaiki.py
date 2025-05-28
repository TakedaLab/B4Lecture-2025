"""Solution to part 1 of ex3 on linear regression."""

import matplotlib.pyplot as plt
import numpy as np


def ols(
    data: np.ndarray,
    order: int,
    reg: str = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> np.ndarray:
    """Ordinary least squares implementation with regularization options.

    Computes simple OLS regression coefficients unless a regularization option is set. If order is
    set, exponential values are computed and added as columns before fitting the data.

    Params:
        data (ndarray): The input data.
        order (int): Order of the output polynomial.
        reg (str): Type of regularization to use optionally ("lasso", "ridge", "elastic_net").
        alpha (float): Regularization parameter.
        l1 (float): l1 ratio to be used in Elastic Net regression.
    Returns:
        w (ndarray): Array of regression coefficients.
    """
    y = data[:, -1][:, np.newaxis]
    x = data[:, :-1]
    # Increase order with exponent terms
    x_raw = np.hstack([x**i for i in range(1, order + 1)])
    x = np.hstack((np.ones(len(x_raw))[:, np.newaxis], x_raw))

    match reg:
        case "lasso":
            # Start with zero weights
            w = np.zeros(x.shape[1])[:, np.newaxis]
            # Experimental: start with random weights
            # w = np.random.uniform(-1, 1, x.shape[1])[:, np.newaxis]

            # Max iterations set to 1000
            for _ in range(1000):
                w_prev = np.copy(w)

                for i, _ in enumerate(w):
                    # For each weight in order
                    x_i = x[:, i]
                    r = np.dot(
                        x_i,
                        y - np.dot(np.delete(x, i, 1), np.delete(w, i))[:, np.newaxis],
                    )
                    # Alternative equation for rho
                    # r = x_i @ (y - x @ w + w[i] * x_i[:, np.newaxis])

                    # Update weight according to soft threshold
                    z_i = np.sum(x_i**2)
                    if r < -alpha:
                        w[i] = (r + alpha) / z_i
                    elif r > alpha:
                        w[i] = (r - alpha) / z_i
                    else:
                        w[i] = 0

                # Stop early if biggest weight update was less than 10e-5
                if np.linalg.norm(w - w_prev, 2) < 10e-5:
                    break
        case "ridge":
            # Ridge regression closed form
            w = np.dot(
                np.dot(
                    np.linalg.inv(np.dot(x.T, x) + alpha * np.identity(x.shape[1])), x.T
                ),
                y,
            )
        case "elastic_net":
            # Start with zero weights
            w = np.zeros(x.shape[1])[:, np.newaxis]
            # Experimental: start with random weights
            # w = np.random.uniform(-1, 1, x.shape[1])[:, np.newaxis]

            # Max iterations set to 1000
            for _ in range(1000):
                w_prev = np.copy(w)

                for i, _ in enumerate(w):
                    # For each weight in order
                    x_i = x[:, i]
                    r = np.dot(
                        x_i,
                        y - np.dot(np.delete(x, i, 1), np.delete(w, i))[:, np.newaxis],
                    )
                    # Alternative equation for rho
                    # r = x_i @ (y - x @ w + w[i] * x_i[:, np.newaxis])

                    # Update weight according to soft threshold

                    # L1 and L2 terms
                    l1 = alpha * l1_ratio
                    l2 = alpha * (1 - l1_ratio)
                    den = np.sum(x_i**2) + l2
                    if r < -l1:
                        w[i] = (r + l1) / den
                    elif r > l1:
                        w[i] = (r - l1) / den
                    else:
                        w[i] = 0

                # Stop early if biggest weight update was less than 10e-5
                if np.linalg.norm(w - w_prev, 2) < 10e-5:
                    break
        case _:
            # OLS Regression closed form
            w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)

    return w


def main() -> None:
    """Execute main routine.

    Performs linear regression on input data (data1.csv-data3.csv) and presents the results using
    matplotlib.

    Params: None
    Returns: None
    """
    # Linear regression with OLS

    # Data 1
    data1 = np.loadtxt("ex3/data/data1.csv", skiprows=1, delimiter=",")
    w1 = ols(data1, order=1, reg="lasso", alpha=0.05)

    # Plots

    # Scatter
    plt.scatter(data1[:, 0], data1[:, 1], 20, "#912583", zorder=2)  # Data points
    plt.title("Data 1 Scatterplot")
    plt.grid()
    plt.savefig("ex3/i_tzimas/data1_scatter.png")

    # With regression line
    plt.title("Data 1 Scatterplot with regression line")
    x = np.array([min(data1[:, 0]), max(data1[:, 0])])
    plt.plot(x, w1[0] + w1[1] * x)
    plt.savefig("ex3/i_tzimas/data1_lr.png")
    plt.clf()

    # Data 2
    data2 = np.loadtxt("ex3/data/data2.csv", skiprows=1, delimiter=",")
    w2 = ols(data2, order=3, reg="elastic_net", alpha=0.05, l1_ratio=0.5)

    # Plots

    # Scatter
    plt.scatter(data2[:, 0], data2[:, 1], 20, "#912583", zorder=2)
    plt.title("Data 2 Scatterplot")
    plt.grid()
    plt.savefig("ex3/i_tzimas/data2_scatter.png")

    # With regression curve
    plt.title("Data 2 Scatterplot with regression curve")
    x = np.linspace(min(data2[:, 0]), max(data2[:, 0]))
    plt.plot(x, w2[0] + w2[1] * x + w2[2] * x**2 + w2[3] * x**3)
    plt.savefig("ex3/i_tzimas/data2_lr.png")
    plt.clf()

    # Data 3
    data3 = np.loadtxt("ex3/data/data3.csv", skiprows=1, delimiter=",")
    w3 = ols(data3, order=2)

    # Plots

    # Scatter
    ax = plt.subplot(111, projection="3d")
    ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], c="#2225c3")
    plt.title("Data 3 Scatterplot")
    # savefig omitted to preserve the already saved image as it has a better viewing angle.
    # plt.savefig("ex3/i_tzimas/data3_scatter.png")

    # With regression surface
    plt.title("Data 3 Scatterplot with regression surface")
    x = np.linspace(min(data3[:, 0]), max(data3[:, 0]))
    y = np.linspace(min(data3[:, 1]), max(data3[:, 1]))
    x, y = np.meshgrid(x, y)
    ax.plot_surface(
        x,
        y,
        w3[0] + w3[1] * x + w3[2] * y + w3[3] * x**2 + w3[4] * y**2,
        color="#33883370",
    )
    # savefig omitted to preserve the already saved image as it has a better viewing angle.
    # plt.savefig("ex3/i_tzimas/data3_lr.png")


if __name__ == "__main__":
    main()
