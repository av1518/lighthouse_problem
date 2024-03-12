import numpy as np
import zeus
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import time
from corner import corner
import matplotlib.ticker as ticker


def log_likelihood(theta, data):
    """
    Calculate the log-likelihood for part (v).

    Args:
    @param theta A tuple or list containing the parameters alpha and beta.
                 - alpha (float): The position of the lighthouse along the coast.
                 - beta (float): The distance of the lighthouse from the shore.
    @param data A numpy array of observed flash locations along the coastline.

    @return The log-likelihood of the observed data given the parameters.
    """
    alpha, beta = theta
    return np.log(beta / np.pi) * len(data) - np.sum(
        np.log(beta**2 + (data - alpha) ** 2)
    )


def log_posterior(
    theta, data, alpha_min=-10, alpha_max=10, beta_min=0.001, beta_max=10
):
    """
    Calculate the log-posterior PDF for part (v).

    @param theta A tuple or list containing the parameters alpha and beta.
            - alpha (float): The displacement of the lighthouse along the coast.
            - beta (float): The displacement of the lighthouse from the shore.
    @param data A numpy array of observed flash locations along the coastline.
    @param alpha_min The minimum value of the uniform pdf prior for alpha. Default is -10.
    @param alpha_max The maximum value of the uniform pdf prior for alpha. Default is 10.
    @param beta_min The minimum value of the uniform pdf prior for beta. Default is 0.01.
    @param beta_max The maximum value of the uniform pdf prior for beta. Default is 10.

    @return The log posterior probability of the observed data given the parameters.
            Returns negative infinity if the parameters are outside their prior range.
            This is to tell zeus to ignore this value.

    """

    alpha, beta = theta
    if beta_min < beta < beta_max and alpha_min < alpha < alpha_max:
        return log_likelihood(theta, data)
    else:
        return -np.inf  # for zeus to ignore this region


def scientific_notation(x, pos):
    if x != 0:
        exponent = int(np.floor(np.log10(np.abs(x))))
        coeff = x / 10**exponent
        return r"${:.0f}\times10^{{{}}}$".format(coeff, exponent)
    else:
        return r"$0$"


def plot_posterior_contours(
    log_posterior_func, data, alpha_range=(-5, 5), beta_range=(0.01, 6), grid_size=200
):
    """
    Plot the posterior distribution of alpha and beta.

    Parameters:
    log_posterior_func (function): The function to compute the log posterior probability.
    data (array-like): The flash location data for the lighthouse problem.
    alpha_range (tuple): The range of values for alpha (min, max).
    beta_range (tuple): The range of values for beta (min, max).
    grid_size (int): The number of points in each dimension of the grid.
    """

    # Define the ranges for alpha and beta
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], grid_size)
    beta_vals = np.linspace(beta_range[0], beta_range[1], grid_size)

    # Create a meshgrid
    alpha_grid, beta_grid = np.meshgrid(alpha_vals, beta_vals)

    # Initialize an array to hold the posterior probabilities
    posterior = np.zeros(alpha_grid.shape)

    # Calculate the posterior probability for each grid point
    for i in range(alpha_grid.shape[0]):
        for j in range(alpha_grid.shape[1]):
            theta = [alpha_grid[i, j], beta_grid[i, j]]
            posterior[i, j] = np.exp(log_posterior_func(theta, data))

    # Plot the results
    plt.figure(figsize=(8, 6), dpi=500)
    contourf = plt.contourf(
        alpha_grid, beta_grid, posterior, levels=100, cmap="viridis"
    )
    cbar = plt.colorbar(contourf, label="Posterior Probability")

    cbar.formatter = ticker.FuncFormatter(scientific_notation)
    cbar.update_ticks()

    # Add contour lines for better visualization
    contour = plt.contour(
        alpha_grid, beta_grid, posterior, levels=contourf.levels[::20], colors="k"
    )
    plt.clabel(contour, inline=True, fontsize=8)

    plt.xlabel(r"$\alpha$", fontsize=15)
    plt.ylabel(r"$\beta$", fontsize=15)
    plt.title("Posterior Distribution of Alpha and Beta")
    plt.xticks(np.arange(alpha_range[0], alpha_range[1], 1))  # Add more xticks
    plt.yticks(np.arange(0, beta_range[1], 0.5))  # Add more yticks
    plt.show()
