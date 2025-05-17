"""Solution to part 2 of ex3 on Principal Component Analysis."""

import matplotlib.pyplot as plt
import numpy as np


def pca(
    data: np.ndarray, output_size: int = None, use_percent: bool = False
) -> np.ndarray:
    """Calculate and return the principal components of input data.

    Principal components calculated as eigenvectors of the covariate matrix. The number of vectors
    to be returned can be specified, or alternatively the function can return the miminum size
    needed to explain 90% of the variance in the data.

    Params:
        data (ndarray): The input data.
        output_size (int): The number of vectors to be returned. All are returned by default.
        use_percent (bool): If set to True, output_size is ignored and instead the minimum
        required size is returned to explain 90% of the variance in the input data.
    Returns:
        pc (ndarray): Array of principal components.
    """
    # Subtract mean for standardization
    data_std = data - np.mean(data, axis=0)
    # Covariate matrix
    data_cov = data_std.T @ data_std / len(data_std)
    # Calculate eigenvalues/vectors
    eig_val, eig_vec = np.linalg.eig(data_cov)
    # Sort based on eigenvalues in descending order (large value = important component)
    sort = np.argsort(eig_val)
    eig_val = eig_val[sort[::-1]]
    pc = eig_vec[:, sort[::-1]]

    if not use_percent:
        # Return all PC or the requested size
        if output_size:
            return pc[:output_size]
        else:
            return pc
    else:
        # Return as many as needed for 90% explained variance
        size = 1
        while np.cumsum([i / np.sum(eig_val) for i in eig_val[:size]]) <= 0.9:
            size += 1
        # Uncomment lines below to print results for data4
        # print(
        #     f"Explained variance: {np.cumsum([i / np.sum(eig_val) for i in eig_val[:size]])}"
        # )
        # print(f"No. of PC: {size}")
        return pc[:size]


def main() -> None:
    """Execute main routine.

    Apples PCA to the given input data (data1.csv-data4.csv) and presents results using matplotlib.

    Params: None
    Returns: None
    """
    # Data 1
    data1 = np.loadtxt("ex3/data/data1.csv", skiprows=1, delimiter=",")
    data1_std = data1 - np.mean(data1, axis=0)
    data1_pc = pca(data1)

    # Plot
    plt.scatter(
        data1_std[:, 0], data1_std[:, 1], 20, "#912583", zorder=2
    )  # Data points
    plt.title("Data 1 Scatterplot with PCA")
    plt.grid()
    for v in data1_pc:
        plt.axline((0, 0), (v[0], v[1]), ls="--")
    plt.savefig("ex3/i_tzimas/data1_pca.png")
    plt.clf()

    # Data 2
    data2 = np.loadtxt("ex3/data/data2.csv", skiprows=1, delimiter=",")
    data2_std = data2 - np.mean(data2, axis=0)
    data2_pc = pca(data2)

    # Plot
    plt.scatter(
        data2_std[:, 0], data2_std[:, 1], 20, "#912583", zorder=2
    )  # Data points
    plt.title("Data 2 Scatterplot with PCA")
    plt.grid()
    for v in data2_pc:
        plt.axline((0, 0), (v[0], v[1]), ls="--")
    plt.savefig("ex3/i_tzimas/data2_pca.png")
    plt.clf()

    # Data 3
    data3 = np.loadtxt("ex3/data/data3.csv", skiprows=1, delimiter=",")
    data3_std = data3 - np.mean(data3, axis=0)
    data3_pc = pca(data3)

    # Plot
    plt.subplot(111, projection="3d").scatter(
        data3_std[:, 0], data3_std[:, 1], data3_std[:, 2]
    )  # Data points
    plt.title("Data 3 Scatterplot with PCA")
    plt.xlim(min(data3_std[:, 0]), max(data3_std[:, 0]))
    plt.ylim(min(data3_std[:, 1]), max(data3_std[:, 1]))
    for i, v in enumerate(data3_pc):
        # Scaling adjusted for data3
        if i == 1:
            v_ext = 80 * v
        else:
            v_ext = 5 * v
        plt.plot([-v_ext[0], v_ext[0]], [-v_ext[1], v_ext[1]], [-v_ext[2], v_ext[2]])
    # savefig omitted to preserve the already saved image as it has a better viewing angle.
    # plt.savefig("ex3/i_tzimas/data3_pca.png")
    plt.clf()

    # Data 3 compression
    data3_pc = pca(data3, 2)
    data3_proj = np.dot(data3, data3_pc.T)

    # Plot
    plt.scatter(
        data3_proj[:, 0], data3_proj[:, 1], 20, "#912583", zorder=2
    )  # Data points
    plt.title("Data 3 Scatterplot (Compressed)")
    plt.grid()
    plt.savefig("ex3/i_tzimas/data3_compressed.png")

    # Data 4
    data4 = np.loadtxt("ex3/data/data4.csv", delimiter=",")
    _ = pca(data4, use_percent=True)


if __name__ == "__main__":
    main()
