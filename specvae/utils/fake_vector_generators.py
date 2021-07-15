#!/usr/bin/env python3

import numpy as np


def random_sine_functions(
    N, M, mu_w=np.pi, sd_w=1.0, mu_A=1.0, sd_A=1.0, mu_phi=0.0, sd_phi=1.0
):
    """Generate an N x M matrix of randomly generated sine functions. Random
    variables will be sampled from Gaussian distributions. The sine function
    is computed on a grid from -2pi -> 2pi, although this grid is not
    meaningful in the machine learning context.

    Parameters
    ----------
    N, M : int
        Specifies the dimension of the returned matrix, N x M.
    mu_w, sd_w : float
        The mean and standard deviation of the frequency.
    mu_A, sd_A : float
        The mean and standard deviation of the amplitude.
    mu_phi, sd_phi : float
        The mean and stanadrd deviation of the phase.

    Returns
    -------
    np.ndarray
        Of shape N x M.
    """

    w = np.random.normal(loc=mu_w, scale=sd_w, size=(N, 1))
    A = np.random.normal(loc=mu_A, scale=sd_A, size=(N, 1))
    phi = np.random.normal(loc=mu_phi, scale=sd_phi, size=(N, 1))
    x = np.linspace(-2.0 * np.pi, 2.0 * np.pi, M).reshape(1, M)
    return np.sin(w * (x + phi)) * A
