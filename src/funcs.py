import numpy as np
import matplotlib.pyplot as plt
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
    log_posterior_func, data, alpha_range=(-2, 2), beta_range=(0.01, 3.5), grid_size=200
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
    # print(contourf.levels[::20])
    contour = plt.contour(
        alpha_grid,
        beta_grid,
        posterior,
        levels=contourf.levels[::20],
        colors="k",
    )
    plt.clabel(contour, inline=True, fontsize=12)

    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel(r"$\beta$", fontsize=20)
    plt.title(r"Unnormalised Posterior Distribution")
    plt.xticks(np.arange(alpha_range[0], alpha_range[1], 1))
    plt.yticks(np.arange(0, beta_range[1], 0.5))
    plt.savefig("figures/posterior_contours_v.png", dpi=300, bbox_inches="tight")
    # plt.show()


def gelman_rubin_for_multiple_chains(chains, frequency):
    """
    @brief Calculates the Gelman-Rubin diagnostic at regular intervals for multiple MCMC chains.

    This function computes the Gelman-Rubin diagnostic for assessing the convergence of
    multiple Markov Chain Monte Carlo (MCMC) chains. It evaluates the diagnostic at
    regular intervals defined by 'frequency' up to the length of the shortest chain.
    The diagnostic is computed by splitting each chain into two sub-chains and using
    these sub-chains to calculate the Gelman-Rubin statistic.

    @param chains A list of MCMC chains, each represented as a list of samples.
                  The chains should all be of the same length.
    @param frequency An integer defining the interval at which the diagnostic should
                     be calculated.

    @return A list of tuples. Each tuple contains two elements: the iteration number
            at which the diagnostic was calculated and the corresponding Gelman-Rubin
            statistic.

    Example usage:
    >>> chains = [chain1, chain2, chain3]  # where chain1, chain2, chain3 are lists
    >>> frequency = 100
    >>> results = gelman_rubin_for_multiple_chains(chains, frequency)

    Note:
    - The function requires a helper function `compute_gelman_rubin(sub_chains)` which
      computes the Gelman-Rubin statistic for a given set of sub-chains.
    - The function assumes that all chains are of equal length and contain numerical values.
    """
    min_length = min(len(chain) for chain in chains)
    results = []

    for i in range(frequency, min_length + 1, frequency):
        # Split each chain and store the sub-chains
        sub_chains = [chain[: i // 2] for chain in chains] + [
            chain[i // 2 : i] for chain in chains
        ]
        # Calculate Gelman-Rubin Diagnostic for the sub-chains
        R = compute_gelman_rubin(sub_chains)
        results.append((i, R))

    return results


def compute_gelman_rubin(sub_chains):
    """
    @brief Computes the Gelman-Rubin diagnostic statistic for a set of sub-chains.

    This function calculates the Gelman-Rubin diagnostic, a convergence diagnostic
    for Markov Chain Monte Carlo (MCMC) simulations. It uses the between-chain variance
    and within-chain variance to compute the R-hat statistic, which is a measure of
    convergence. An R-hat value close to 1 indicates that the chains have converged.

    The function assumes that all sub-chains are of equal length and contains numerical
    values.

    @param sub_chains A list of sub-chains, each represented as a list of samples.
                      These sub-chains are used to calculate the diagnostic.

    @return The computed Gelman-Rubin R-hat statistic. A value close to 1 indicates
            convergence of the chains.

    Example usage:
    >>> sub_chains = [chain1[:len(chain1)//2], chain1[len(chain1)//2:], chain2[:len(chain2)//2], chain2[len(chain2)//2:]]
    >>> R = compute_gelman_rubin(sub_chains)

    Note:
    - The function uses numpy for calculating means and variances.
    - 'np.mean' is used for calculating the mean of each sub-chain.
    - 'np.var' with 'ddof=1' is used for calculating the sample variance of each sub-chain.
    """

    means = [np.mean(chain) for chain in sub_chains]
    n = len(sub_chains[0])
    variances = [np.var(chain, ddof=1) for chain in sub_chains]
    W = np.mean(variances)
    B = n * np.var(means, ddof=1)
    # Assuming all sub-chains are of equal length
    Var_plus = (n - 1) / n * W + B / n
    R = np.sqrt(Var_plus / W)
    return R


def log_likelihood_loc(theta, data):
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


def log_likelihood_int(alpha, beta, I0, x_data, I_data, sigma=1.0):
    """
    @brief Computes the log-likelihood for including the intensity measurements

    @param alpha The position of the lighthouse along the shore.
    @param beta The distance of the lighthouse from the shore.
    @param I0 The absolute intensity of the lighthouse.
    @param x_data A numpy array containing the positions of detected flashes along the shore.
    @param I_data A numpy array containing the observed intensities of the flashes.
    @param sigma The standard deviation of the Gaussian distribution of log-intensities.
                 Default is 1.0.

    @return The log-likelihood of the observed intensity data given the model parameters.
    """
    # calculate squared distatnce from the lighthouse to each detected flash
    d_squared = beta * beta + (x_data - alpha) ** 2

    # expected log-intensity foe each measurement
    mu = np.log(I0) - np.log(d_squared)

    log_likelihood = -1 / (2 * sigma**2) * np.sum((np.log(I_data) - mu) ** 2) - len(
        I_data
    ) * np.log(np.sqrt(2 * np.pi) * sigma)
    return log_likelihood


def combined_log_likelihood(theta, x_data, I_data):
    """
    @brief Adds the log-likelihoods of the location and intensity data

    @param theta A tuple of parameters (alpha, beta, I0).
    @param x_data A numpy array containing the positions of detected flashes along the shore.
    @param I_data A numpy array containing the observed intensities of the flashes.

    @return The combined log-likelihood of the location and intensity data.
    """

    alpha, beta, I0 = theta
    loc_likelihood = log_likelihood_loc((alpha, beta), x_data)
    int_likelihood = log_likelihood_int(alpha, beta, I0, x_data, I_data)
    return loc_likelihood + int_likelihood


def log_posterior_combined(
    theta,
    loc_data,
    int_data,
    alpha_min=-10,
    alpha_max=10,
    beta_min=0.01,
    beta_max=10,
    I0_min=0.01,
    I0_max=100,
):
    """
    @brief Computes the log-posterior for combined location and intensity data.

    This function calculates the log-posterior probability of the lighthouse model parameters
    (alpha, beta, I0) given location and intensity data. It uses a combined log-likelihood
    function (`combined_log_likelihood`) and applies a Jeffrey's prior for I0. The function
    only computes the log-posterior if the parameters fall within specified ranges;
    otherwise, it returns negative infinity. This is the expected input for the MCMC sampler.

    @param theta A tuple of model parameters (alpha, beta, I0).
    @param loc_data A numpy array containing the positions of detected flashes along the shore.
    @param int_data A numpy array containing the observed intensities of the flashes.
    @param alpha_min The minimum allowed value for alpha. Default is -10.
    @param alpha_max The maximum allowed value for alpha. Default is 10.
    @param beta_min The minimum allowed value for beta. Default is 0.01.
    @param beta_max The maximum allowed value for beta. Default is 10.
    @param I0_min The minimum allowed value for I0. Default is 0.01.
    @param I0_max The maximum allowed value for I0. Default is 100.

    @return The log-posterior probability of the parameters if they fall within the specified ranges.
            If the parameters are outside the ranges, it returns negative infinity.

    """
    alpha, beta, I0 = theta

    if (
        beta_min < beta < beta_max
        and alpha_min < alpha < alpha_max
        and I0_min < I0 < I0_max
    ):
        combined_likelihood = combined_log_likelihood(theta, loc_data, int_data)
        jeffrey_prior = np.log(1 / (I0 * np.log(I0_max / I0_min)))
        return combined_likelihood + jeffrey_prior
    else:
        return -np.inf
