# %%
import numpy as np
import zeus
import matplotlib.pyplot as plt
import pandas as pd
from corner import corner
from matplotlib import rcParams


# Set matplotlib params
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]  # or 'DejaVu Serif'
plt.rcParams["text.usetex"] = False  # Keep as False if LaTeX is not installed
plt.rcParams["mathtext.fontset"] = "stix"  # LaTeX-like fonts for math expressions
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7


from funcs import (
    log_posterior_combined,
    plot_posterior_vs_intensity,
    gelman_rubin_for_multiple_chains,
)

# Update Matplotlib settings to use LaTeX-like font
rcParams["font.family"] = "serif"
rcParams["text.usetex"] = False  # Keep as False if LaTeX is not installed
# load the data
file_path = "../lighthouse_flash_data.txt"
data = pd.read_csv(file_path, sep=" ", header=None)
flash_locations, intensities = data[0].values, data[1].values
# %% Plot the posterior distribution vs. intensity for the best alpha and beta from the previous part
from matplotlib.ticker import FuncFormatter


def scientific_notation(x, pos):
    if x != 0:
        exponent = int(np.floor(np.log10(np.abs(x))))
        coeff = x / 10**exponent
        return r"${:.0f}\times10^{{{}}}$".format(coeff, exponent)
    else:
        return r"$0$"


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
    plt.plot(
        I0_vals,
        posterior,
        label=r" Posterior with fixed $\alpha$ and $\beta$",
        color="blue",
    )
    plt.xlabel(r"$I_0$", fontsize=15)
    plt.ylabel(
        r"$\mathcal{L}(\{ log(I_k) | \hat{\alpha}, \hat{\beta}, I_0)   $ ", fontsize=15
    )
    # plt.title("Posterior Distribution vs. Intensity (I0)")

    # Set the y-axis formatter to scientific notation
    plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))

    plt.legend()
    plt.savefig("posterior_vs_intensity.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_posterior_vs_intensity(log_posterior_combined, flash_locations, intensities)
# %% MCMC sampling
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
sampler_2 = zeus.EnsembleSampler(
    nwalkers, ndim, log_posterior_combined, args=[flash_locations, intensities]
)
# saplter2 = zeus.EnsembleSaplter(nwalkers, ndim, log_posterior, args=[flash_locations])
print("Running first sampler...")
sampler.run_mcmc(start_positions, nsteps)
print("Running second sampler (for use in diagnostics)...")
sampler_2.run_mcmc(start_positions, nsteps)
# %% Autocorrelation time
taus = zeus.AutoCorrTime(sampler.get_chain())
print("AutoCorrelation time :")
print("tau_alpha = {}, tau_beta = {}, tau_I0 = {}".format(*taus))
tau = max(taus)
print(f"Max {tau = }")
# %% Gelman rubin test
chain_1 = sampler.get_chain(flat=True)
chain_2 = sampler_2.get_chain(flat=True)
print(len(chain_1), len(chain_2))
# %% Visualize the chains

fig, axs = plt.subplots(nrows=3, figsize=(7, 10))

for i in range(3):
    axs[0].plot(sampler.get_chain()[:, i, 0][:500], alpha=0.5, label=f"Walker {i+1}")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel(r"$\alpha$")
axs[0].legend()

for i in range(3):
    axs[1].plot(sampler.get_chain()[:, i, 1][:500], alpha=0.5, label=f"Walker {i+1}")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel(r"$\beta$")
axs[1].legend()

for i in range(3):
    axs[2].plot(sampler.get_chain()[:, i, 2][:500], alpha=0.5, label=f"Walker {i+1}")
axs[2].set_xlabel("Iterations")
axs[2].set_ylabel(r"$I_0$")
axs[2].legend()

plt.tight_layout()
plt.savefig("chain_plots.png", dpi=500, bbox_inches="tight")
plt.show()

# %%
alpha_chains_1 = [chain_1[:, 0], chain_2[:, 0]]
beta_chains_1 = [chain_1[:, 1], chain_2[:, 1]]
I0_chains_1 = [chain_1[:, 2], chain_2[:, 2]]

GR_alpha = gelman_rubin_for_multiple_chains(alpha_chains_1, frequency=100)
GR_beta = gelman_rubin_for_multiple_chains(beta_chains_1, frequency=100)
GR_I0 = gelman_rubin_for_multiple_chains(I0_chains_1, frequency=100)


def plot_gelman_rubin(gr_stats_alpha, gr_stats_beta, gr_stats_I0, max_iteration):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), dpi=500)

    # Alpha subplot
    iterations, R_values = zip(*gr_stats_alpha)
    axes[0].plot(
        iterations[:max_iteration], R_values[:max_iteration], marker="o", color="black"
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\hat{R}(\alpha)$ ")
    axes[0].grid(True)

    # Beta subplot
    iterations, R_values = zip(*gr_stats_beta)
    axes[1].plot(
        iterations[:max_iteration], R_values[:max_iteration], marker="o", color="black"
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\hat{R}(\beta)$ ")
    axes[1].grid(True)

    # I0 subplot
    iterations, R_values = zip(*gr_stats_I0)
    axes[2].plot(
        iterations[:max_iteration], R_values[:max_iteration], marker="o", color="black"
    )
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel(r"$\hat{R}(I_0)$")
    axes[2].grid(True)

    # Set a single, overarching title
    fig.suptitle("Gelman-Rubin Diagnostics", fontsize=16)

    fig.subplots_adjust(hspace=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("Gelman-Rubin_Diagnostics.png", bbox_inches="tight")
    plt.show()


# Example usage
plot_gelman_rubin(GR_alpha, GR_beta, GR_I0, 100)  # Plots up to the 100th iteration

# %%
chain_iid = sampler.get_chain(
    flat=True, thin=int(tau), discard=0.02
)  # discard 2% of the chain (burn-in)
print("Number of iid samples after discard burnin and thinning =", len(chain_iid))


alpha_mean, beta_mean, I0_mean = np.mean(chain_iid, axis=0)
alpha_std, beta_std, I0_std = np.std(chain_iid, axis=0)
print(f"Alpha: Mean = {alpha_mean}, Std = {alpha_std}")
print(f"Beta: Mean = {beta_mean}, Std = {beta_std}")
print(f"I0: Mean = {I0_mean}, Std = {I0_std}")

# save the means and stds to a file
with open("means_stds_vii.txt", "w") as file:
    file.write(
        f"Alpha: Mean = {alpha_mean}, Std = {alpha_std}\n"
        f"Beta: Mean = {beta_mean}, Std = {beta_std}\n"
        f"I0: Mean = {I0_mean}, Std = {I0_std}"
    )

corner(
    chain_iid,
    labels=[r"$\alpha$", r"$\beta$", r"$I_0$"],
    truths=[alpha_mean, beta_mean, I0_mean],
    show_titles=True,
    bins=40,
    color="black",
)
plt.savefig("corner_plot_vii.png", dpi=500, bbox_inches="tight")
plt.show()
