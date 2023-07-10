import numpy as np
import matplotlib.pyplot as plt


def kde(samples, h):
    """
    Estimates the density from the samples using Kernel Density Estimation
    :param samples: DxN matrix of data points
    :param h: (half) window size/radius of kernel
    :return:
        est_density: estimated density in the range of [-5,5]
    """
    # Compute the number of samples created
    n = len(samples)

    # Create a linearly spaced vector
    pos = np.arange(-5, 5.0, 0.1)

    # Estimate the density from the samples using a kernel density estimator
    norm = np.sqrt(2 * np.pi) * h * n
    res = np.sum(np.exp(-(pos[np.newaxis, :] - samples[:, np.newaxis]) ** 2 / (2 * h ** 2)), axis=0) / norm

    # Form the output variable
    est_density = np.stack((pos, res), axis=1)

    return est_density


def gauss1D(m, v, N, w):
    """
    Computes the Normal distribution from the samples.
    :param m: Sample mean.
    :param v: Sample variance.
    :param N: Number of sample datapoints.
    :param w: One sided range of sample datapoints.
    :return:
        real_density: Normal distribution obtained from the sample datapoints.
    """
    pos = np.arange(-w, w - w / N, 2 * w / N)
    insE = -0.5 * ((pos - m) / v) ** 2
    norm = 1 / (v * np.sqrt(2 * np.pi))
    res = norm * np.exp(insE)
    real_density = np.stack((pos, res), axis=1)
    return real_density


# Get the parameter
h = 0.3

# Produce the random samples
samples = np.random.normal(0, 1, 100)

# Compute the original normal distribution
real_density = gauss1D(0, 1, 100, 5)

# Estimate the probability density using the KDE
est_density = kde(samples, h)

# plot results
plt.subplot(2, 1, 1)
plt.plot(est_density[:, 0], est_density[:, 1], 'r', linewidth=1.5, label='KDE Estimated Distribution')
plt.plot(real_density[:, 0], real_density[:, 1], 'b', linewidth=1.5, label='Real Distribution')
plt.legend()
plt.show()
