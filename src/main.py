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


from funcs import plot_posterior_contours, log_likelihood, log_posterior


# Update Matplotlib settings to use LaTeX-like font
# Set Matplotlib to use a commonly available serif font
rcParams["font.family"] = "serif"
rcParams["text.usetex"] = False  # Keep as False if LaTeX is not installed
# %%
# load the data
file_path = "../lighthouse_flash_data.txt"
data = pd.read_csv(file_path, sep=" ", header=None)
flash_locations = data[0].values

plot_posterior_contours(log_posterior, flash_locations)
# %%
nwalkers = 50
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

sampler.run_mcmc(start_positions, nsteps, callbacks=[gelman_rub, min_iter])

# %%
taus = zeus.AutoCorrTime(sampler.get_chain())
print("Autocorrelation:", taus)

tau = max(taus)
print(f"{tau = }")

R_diag = gelman_rub.estimates
plt.plot(np.arange(len(R_diag)), R_diag, lw=2.5)
plt.title("Split-R Gelman-Rubin Statistic", fontsize=14)
plt.xlabel("Iterations", fontsize=14)
plt.ylabel(r"$R$", fontsize=14)


# %%
iid = sampler.get_chain(flat=True, thin=int(tau), discard=0.2)
num_samples = len(iid)
print(f"{num_samples = }")

alpha_mean, beta_mean = np.mean(iid, axis=0)
alpha_std, beta_std = np.std(iid, axis=0)
print(f"Alpha: Mean = {alpha_mean}, Std = {alpha_std}")
print(f"Beta: Mean = {beta_mean}, Std = {beta_std}")

# %%
fig, axs = plt.subplots(nrows=2, figsize=(5, 4))

axs[0].plot(sampler.get_chain()[:, :, 0], alpha=0.25)
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel(r"$\alpha$")
axs[0].set_xlim(0, nsteps)

axs[1].plot(sampler.get_chain()[:, :, 1], alpha=0.25)
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel(r"$\beta$")
axs[1].set_xlim(0, nsteps)

plt.tight_layout()
plt.show()
# %% Potting the last few iterations
fig, axs = plt.subplots(nrows=2, figsize=(5, 4))

starting_point = 500

# Assuming nsteps is the total number of iterations
start = nsteps - starting_point

# Retrieve the last 1000 iterations for each parameter
last_alpha = sampler.get_chain()[-starting_point:, :, 0]
last_beta = sampler.get_chain()[-starting_point:, :, 1]

# Plot alpha values for the last 1000 iterations
for idx, walker in enumerate(last_alpha.T):
    if idx == 0:  # Label only the first walker for clarity
        axs[0].plot(range(start, nsteps), walker, alpha=0.5, label=f"Walker {idx}")
    else:
        axs[0].plot(range(start, nsteps), walker, alpha=0.5)
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel(r"$\alpha$")
axs[0].legend()  # Add legend to the plot

# Plot beta values for the last 1000 iterations
for idx, walker in enumerate(last_beta.T):
    if idx == 0:  # Label only the first walker for clarity
        axs[1].plot(range(start, nsteps), walker, alpha=0.5, label=f"Walker {idx}")
    else:
        axs[1].plot(range(start, nsteps), walker, alpha=0.5)
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel(r"$\beta$")
axs[1].legend()  # Add legend to the plot

plt.tight_layout()
plt.show()


# %%
# corner plot
corner(iid, labels=[r"$\alpha$", r"$\beta$"], truths=[alpha_mean, beta_mean])
plt.show()

# %%
plot_posterior_contours(log_posterior, flash_locations)

# %% part (vii)
