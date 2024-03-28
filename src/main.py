# %%
import numpy as np
import zeus
import matplotlib.pyplot as plt
import pandas as pd
from corner import corner

import matplotlib

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
    log_posterior_combined,
    scientific_notation,
)

# %%
# load the data
file_path = "lighthouse_flash_data.txt"
data = pd.read_csv(file_path, sep=" ", header=None)
flash_locations = data[0].values

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


print("Defining the two independent samplers for part (v)")
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
# print(len(chain_1), len(chain_2))
# %%
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
plt.savefig("figures/chain_plots_v.png", dpi=500, bbox_inches="tight")
# plt.savefig()

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
    # fig.suptitle("Gelman-Rubin Diagnostics for part (v)", fontsize=16)

    fig.subplots_adjust(hspace=0.4)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("figures/Gelman-Rubin_Diagnostics_v.png", bbox_inches="tight", dpi=400)
    # plt.show()


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
figure.savefig("figures/corner_plot_v.png", dpi=500, bbox_inches="tight")
# plt.show()


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
plt.savefig("figures/marginal_posteriors_v.png", dpi=500, bbox_inches="tight")
# plt.show()


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
plt.savefig("figures/joint_distribution_v.png", dpi=500, bbox_inches="tight")
# plt.show()

print("All plots for part (v) saved in /figures folder")

# %% PART VII:
# load the data
file_path = "lighthouse_flash_data.txt"
data = pd.read_csv(file_path, sep=" ", header=None)
flash_locations, intensities = data[0].values, data[1].values
# %% Plot the posterior distribution vs. intensity for the best alpha and beta from the previous part
from matplotlib.ticker import FuncFormatter


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
    plt.ylabel(r"$P( I_0 | \hat{\alpha}, \hat{\beta}, {x_k}, {I_k})$ ", fontsize=15)
    # plt.title("Posterior Distribution vs. Intensity (I0)")

    # Set the y-axis formatter to scientific notation
    plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))

    plt.legend()
    plt.savefig("figures/posterior_vs_intensity.png", dpi=300, bbox_inches="tight")
    # plt.show()


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

print("Define two independent samplers for part vii")

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
plt.savefig("figures/chain_plots.png", dpi=500, bbox_inches="tight")
# plt.show()

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
    # fig.suptitle("Gelman-Rubin Diagnostics", fontsize=16)

    fig.subplots_adjust(hspace=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("figures/Gelman-Rubin_Diagnostics.png", bbox_inches="tight")
    # plt.show()


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
plt.savefig("figures/corner_plot_vii.png", dpi=500, bbox_inches="tight")
# plt.show()
print("Plots for part vii saved in /figures")
