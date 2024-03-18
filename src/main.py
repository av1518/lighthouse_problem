# %%
import numpy as np
import zeus
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import time
from corner import corner

import matplotlib
import matplotlib.ticker as ticker

matplotlib.rcParams.update(
    {
        "font.size": 14,
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 14,
        "figure.autolayout": True,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    }
)


from funcs import (
    plot_posterior_contours,
    log_posterior,
    gelman_rubin_for_multiple_chains,
)

# %%
# load the data
file_path = "../lighthouse_flash_data.txt"
data = pd.read_csv(file_path, sep=" ", header=None)
flash_locations = data[0].values


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
    print(contourf.levels[::20])
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
    plt.savefig("posterior_contours_v.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_posterior_contours(log_posterior, flash_locations)
# %%
nwalkers = 10
nsteps = 10000
ndim = 2

alpha_bounds = [-2, 1]
beta_bounds = [1, 3]

# Generating initial positions within the bounds
start_positions = np.zeros((nwalkers, ndim))
start_positions[:, 0] = np.random.uniform(
    alpha_bounds[0], alpha_bounds[1], nwalkers
)  # Alpha
start_positions[:, 1] = np.random.uniform(
    beta_bounds[0], beta_bounds[1], nwalkers
)  # Beta

gelman_rub = zeus.callbacks.SplitRCallback(
    ncheck=200, epsilon=0.01, nsplits=10, discard=0.2
)

min_iter = zeus.callbacks.MinIterCallback(nmin=10000)

sampler = zeus.EnsembleSampler(nwalkers, ndim, log_posterior, args=[flash_locations])
sampler_2 = zeus.EnsembleSampler(nwalkers, ndim, log_posterior, args=[flash_locations])

sampler.run_mcmc(start_positions, nsteps)
sampler_2.run_mcmc(start_positions, nsteps)

# %%
taus = zeus.AutoCorrTime(sampler.get_chain())
print("AutoCorrelation time :")
print("tau_alpha = {}, tau_beta = {}".format(*taus))
tau = max(taus)
print(f"Max {tau = }")
# %% Gelman rubin test
chain_1 = sampler.get_chain(flat=True)
chain_2 = sampler_2.get_chain(flat=True)
print(len(chain_1), len(chain_2))

fig, axs = plt.subplots(nrows=2, figsize=(7, 7))

# Plotting for parameter alpha
for i in range(2):
    axs[0].plot(sampler.get_chain()[:, i, 0][:500], alpha=0.5, label=f"Walker {i+1}")

# Plot initial positions of all walkers for alpha
initial_positions_alpha = sampler.get_chain()[0, :, 0]
axs[0].scatter(
    [0] * len(initial_positions_alpha),
    initial_positions_alpha,
    color="black",
    marker="x",
    zorder=5,
    label="Starting values",
)

axs[0].set_xlabel("Iterations", fontsize=16)
axs[0].set_ylabel(r"$\alpha$", fontsize=20)
axs[0].legend(loc="upper right")

# Plotting for parameter beta
for i in range(2):
    axs[1].plot(sampler.get_chain()[:, i, 1][:500], alpha=0.5, label=f"Walker {i+1}")

# Plot initial positions of all walkers for beta
initial_positions_beta = sampler.get_chain()[0, :, 1]
axs[1].scatter(
    [0] * len(initial_positions_beta),
    initial_positions_beta,
    color="black",
    marker="x",
    zorder=5,
    label="Starting values",
)

axs[1].set_xlabel("Iterations", fontsize=16)
axs[1].set_ylabel(r"$\beta$", fontsize=20)
axs[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("chain_plots_v.png", dpi=500, bbox_inches="tight")
plt.show()

# %%
alpha_chains_1 = [chain_1[:, 0], chain_2[:, 0]]
beta_chains_1 = [chain_1[:, 1], chain_2[:, 1]]

GR_alpha = gelman_rubin_for_multiple_chains(alpha_chains_1, frequency=100)
GR_beta = gelman_rubin_for_multiple_chains(beta_chains_1, frequency=100)


def plot_gelman_rubin(gr_stats_alpha, gr_stats_beta, max_iteration):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=500)

    # Alpha subplot
    iterations, R_values = zip(*gr_stats_alpha)
    axes[0].plot(
        iterations[:max_iteration], R_values[:max_iteration], marker="o", color="navy"
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\hat{R}(\alpha)$ ", fontsize=20)
    axes[0].grid(True)

    # Beta subplot
    iterations, R_values = zip(*gr_stats_beta)
    axes[1].plot(
        iterations[:max_iteration], R_values[:max_iteration], marker="o", color="navy"
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\hat{R}(\beta)$ ", fontsize=20)
    axes[1].grid(True)

    # Set a single  title
    fig.suptitle("Gelman-Rubin Diagnostics for part (v)", fontsize=16)

    fig.subplots_adjust(hspace=0.4)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("Gelman-Rubin_Diagnostics_v.png", bbox_inches="tight", dpi=400)
    plt.show()


plot_gelman_rubin(GR_alpha, GR_beta, 100)  # Plots up to the 100th iteration


# %%
chain_iid = sampler.get_chain(
    flat=True, thin=int(tau), discard=0.02
)  # discard 2% of the chain (burn-in)
print("Number of iid samples after discard burnin and thinning =", len(chain_iid))

# %% Corner plot
alpha_mean, beta_mean = np.mean(chain_iid, axis=0)
alpha_std, beta_std = np.std(chain_iid, axis=0)
print(f"Alpha: Mean = {alpha_mean}, Std = {alpha_std}")
print(f"Beta: Mean = {beta_mean}, Std = {beta_std}")

# save the means and stds to a file
with open("means_stds_v.txt", "w") as file:
    file.write(
        f"Alpha: Mean = {alpha_mean}, Std = {alpha_std}\n"
        f"Beta: Mean = {beta_mean}, Std = {beta_std}\n"
    )

# Generate the corner plot with a customized heatmap
figure = corner(
    chain_iid,
    labels=[r"$\alpha$", r"$\beta$"],
    truths=[alpha_mean, beta_mean],
    bins=50,
    fig=plt.figure(figsize=(8, 8), dpi=500),
    label_kwargs={"fontsize": 20},
    hist2d_kwargs={"bins": 50, "cmap": "viridis"},
)

titles = [
    r"Marginalized Posterior of $\alpha$",
    "",
    "Joint Posterior",
    r"Marginalized Posterior of $\beta$",
]
for ax, title in zip(figure.axes, titles):
    ax.set_title(title, fontsize=14)

# Adding y-axis labels and mean/std info to the diagonal plots
figure.axes[0].set_ylabel("Frequency", fontsize=14)  # For alpha
figure.axes[0].text(
    0.02,
    0.9,
    r"$\hat{\alpha} = -0.45 \pm 0.60 $",
    transform=figure.axes[0].transAxes,
    fontsize=12,
)

figure.axes[-1].set_ylabel("Frequency", fontsize=14)  # For beta
figure.axes[-1].text(
    0.25,
    0.9,
    r"$\hat{\beta} = 1.96 \pm 0.67 $",
    transform=figure.axes[-1].transAxes,
    fontsize=12,
)
figure.savefig("corner_plot_v.png", dpi=500, bbox_inches="tight")
plt.show()


# %% 1-d histograms
# Extract alpha and beta samples
alpha_samples = chain_iid[:, 0]
beta_samples = chain_iid[:, 1]

# Calculate means
alpha_mean = np.mean(alpha_samples)
beta_mean = np.mean(beta_samples)

# Set up the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=500)

# Plot histogram for alpha
axs[0].hist(alpha_samples, bins=50, color="gray")
axs[0].set_title(r"Marginalized Posterior of $\alpha$")
axs[0].set_xlabel(r"$\alpha$", fontsize=20)
axs[0].set_ylabel("Frequency", fontsize=16)


# Add vertical line at mean for alpha
axs[0].axvline(
    alpha_mean,
    color="navy",
    linestyle="dashed",
    linewidth=1,
    label=r"$\hat{\alpha} = -0.45 \pm 0.60 $",
)
axs[0].legend()


# Plot histogram for beta
axs[1].hist(beta_samples, bins=50, color="gray")
axs[1].set_title(r"Marginalized Posterior of $\beta$")
axs[1].set_xlabel(r"$\beta$", fontsize=20)
axs[1].set_ylabel("Frequency", fontsize=16)
# Add vertical line at mean for beta
axs[1].axvline(
    beta_mean,
    color="navy",
    linestyle="dashed",
    linewidth=1,
    label=r"$\hat{\beta} = 1.96 \pm 0.67 $",
)

plt.legend()
# Adjust the layout and display the plot
plt.tight_layout()
plt.savefig("marginal_posteriors_v.png", dpi=500, bbox_inches="tight")
plt.show()


# %% 2-d joint histogram
# Assuming chain_iid contains your samples with the first column as alpha and the second as beta
alpha_samples = chain_iid[:, 0]
beta_samples = chain_iid[:, 1]

# Plotting the 2D histogram / heatmap
plt.figure(figsize=(7, 6), dpi=500)
plt.hist2d(alpha_samples, beta_samples, bins=50, cmap="viridis")
plt.colorbar(label="Density")
plt.xlabel(r"$\alpha$", fontsize=20)
plt.ylabel(r"$\beta$", fontsize=20)
plt.title("Joint Distribution")
plt.savefig("joint_distribution_v.png", dpi=500, bbox_inches="tight")
plt.show()
