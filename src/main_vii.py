# %%
import numpy as np
import zeus
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import time
from corner import corner

from matplotlib import rcParams
import matplotlib.ticker as ticker


from funcs import (
    plot_posterior_contours,
    log_likelihood,
    log_posterior,
    gelman_rubin_for_multiple_chains,
)


# Update Matplotlib settings to use LaTeX-like font
# Set Matplotlib to use a commonly available serif font
rcParams["font.family"] = "serif"
rcParams["text.usetex"] = False  # Keep as False if LaTeX is not installed
# %%
# load the data
file_path = "../lighthouse_flash_data.txt"
data = pd.read_csv(file_path, sep=" ", header=None)
flash_locations, intensities = data[0].values, data[1].values


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
    # calculate squared distatnce from the lighthouse to each detected flash
    d_squared = beta * beta + (x_data - alpha) ** 2

    # expected log-intensity foe each measurement
    mu = np.log(I0) - np.log(d_squared)

    log_likelihood = -1 / (2 * sigma**2) * np.sum((np.log(I_data) - mu) ** 2) - len(
        I_data
    ) * np.log(np.sqrt(2 * np.pi) * sigma)
    return log_likelihood


def combined_log_likelihood(theta, x_data, I_data):
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
    alpha, beta, I0 = theta

    if (
        beta_min < beta < beta_max
        and alpha_min < alpha < alpha_max
        and I0_min < I0 < I0_max
    ):
        combined_likelihood = combined_log_likelihood(theta, loc_data, int_data)
        # log_prior_I0 = -np.log(sigma_I0 * np.sqrt(2 * np.pi)) - (
        #     np.log(I0) - mu_I0
        # ) ** 2 / (2 * sigma_I0**2)
        jeffrey_prior = np.log(1 / (I0 * np.log(I0_max / I0_min)))
        return combined_likelihood + jeffrey_prior
    else:
        return -np.inf


def plot_posterior_vs_intensity(
    log_posterior_func,
    loc_data,
    int_data,
    alpha_best=-0.4545,
    beta_best=1.9705,
    I0_range=(0.01, 10),
    grid_size=200,
):
    I0_vals = np.linspace(I0_range[0], I0_range[1], grid_size)

    # Initialize an array to hold the posterior probabilities
    posterior = np.zeros(grid_size)

    # Calculate the posterior probability for each value of I0
    for i, I0 in enumerate(I0_vals):
        theta = [alpha_best, beta_best, I0]
        posterior[i] = np.exp(log_posterior_func(theta, loc_data, int_data))

    # Plot the results
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(I0_vals, posterior, label="Posterior vs. I0")
    plt.xlabel(r"$I_0$", fontsize=15)
    plt.ylabel("Posterior Probability", fontsize=15)
    plt.title("Posterior Distribution vs. Intensity (I0)")
    plt.legend()
    plt.show()


# %%
plot_posterior_vs_intensity(log_posterior_combined, flash_locations, intensities)
# %%
nwalkers = 10
nsteps = 10000
ndim = 3

alpha_start_bounds = [-5, 5]
beta_start_bounds = [0.02, 5]
I0_start_bounds = [2, 8]


# Generating initial positions within the bounds
start_positions = np.zeros((nwalkers, ndim))
start_positions[:, 0] = np.random.uniform(
    alpha_start_bounds[0], alpha_start_bounds[1], nwalkers
)  # Alpha
start_positions[:, 1] = np.random.uniform(
    beta_start_bounds[0], beta_start_bounds[1], nwalkers
)  # Beta
start_positions[:, 2] = np.random.uniform(
    I0_start_bounds[0], I0_start_bounds[1], nwalkers
)  # I0


sampler = zeus.EnsembleSampler(
    nwalkers, ndim, log_posterior_combined, args=[flash_locations, intensities]
)
# sampler2 = zeus.EnsembleSampler(nwalkers, ndim, log_posterior, args=[flash_locations])
sampler.run_mcmc(start_positions, nsteps)
# sampler2.run_mcmc(start_positions, nsteps)

# %%
taus = zeus.AutoCorrTime(sampler.get_chain())
print("Autocorrelation:", taus)

tau = max(taus)
print(f"{tau = }")
# %% Gelman rubin test
iid = sampler.get_chain(flat=True)
num_samples = len(iid)
print(f"{num_samples = }")

# iid_2 = sampler2.get_chain(flat=True, thin=int(tau), discard=0.2)

# %%
# alpha_chains = [iid[:, 0], iid_2[:, 0]]
# beta_chains = [iid[:, 1], iid_2[:, 1]]

alpha_mean, beta_mean, I0_mean = np.mean(iid, axis=0)
alpha_std, beta_std, I0_std = np.std(iid, axis=0)
print(f"Alpha: Mean = {alpha_mean}, Std = {alpha_std}")
print(f"Beta: Mean = {beta_mean}, Std = {beta_std}")
print(f"I0: Mean = {I0_mean}, Std = {I0_std}")

corner(
    iid,
    labels=[r"$\alpha$", r"$\beta$", r"$I_0$"],
    truths=[alpha_mean, beta_mean, I0_mean],
)
plt.show()
# %%

GR_alpha = gelman_rubin_for_multiple_chains(alpha_chains, frequency=10)
GR_beta = gelman_rubin_for_multiple_chains(beta_chains, frequency=10)
