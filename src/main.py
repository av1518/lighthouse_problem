# %%
import numpy as np
import zeus
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import time
from corner import corner

# %%
# load the data
file_path = "../lighthouse_flash_data.txt"
data = pd.read_csv(file_path, sep=" ", header=None)
flash_locations = data[0].values


# %%
def log_likelihood(theta, data):
    alpha, beta = theta
    # ll_sum = 0

    # for x in data:
    #     ll_sum += np.log(beta / np.pi) - 2 * np.log(beta**2 + (x - alpha) ** 2)

    return np.log(beta / np.pi) * len(data) - np.sum(
        np.log(beta**2 + (data - alpha) ** 2)
    )
    # return ll_sum


def log_posterior(theta, data):
    alpha, beta = theta
    alpha_min = -10
    alpha_max = 10
    if beta > 0.01 and alpha_min < alpha < alpha_max:
        return log_likelihood(theta, data)
    else:
        return -np.inf  # for zeus to ignore this region


nwalkers = 10
nsteps = 100000
ndim = 2

alpha_bounds = [-10, 10]
beta_bounds = [1, 5]

# Generating initial positions within the bounds
start_positions = np.zeros((nwalkers, ndim))
start_positions[:, 0] = np.random.uniform(
    alpha_bounds[0], alpha_bounds[1], nwalkers
)  # Alpha
start_positions[:, 1] = np.random.uniform(
    beta_bounds[0], beta_bounds[1], nwalkers
)  # Beta

sampler = zeus.EnsembleSampler(nwalkers, ndim, log_posterior, args=[flash_locations])

sampler.run_mcmc(start_positions, nsteps)

# %%
taus = zeus.AutoCorrTime(sampler.get_chain())
print("Autocorrelation:", taus)

tau = max(taus)
print(f"{tau = }")

# %%
burnin = 1000
iid_samples = sampler.get_chain(flat=True, thin=int(tau), discard=0.05)
iid = iid_samples[burnin:]
num_samples = len(iid)
print(f"{num_samples = }")

alpha_mean, beta_mean = np.mean(iid[burnin:], axis=0)
alpha_std, beta_std = np.std(iid[burnin:], axis=0)
print(f"Alpha: Mean = {alpha_mean}, Std = {alpha_std}")
print(f"Beta: Mean = {beta_mean}, Std = {beta_std}")

# %%
fig, axs = plt.subplots(nrows=ndim, figsize=(5, ndim * 2))

for i, ax in enumerate(axs):
    ax.plot(sampler.get_chain()[:, :, i], alpha=0.25)
    # ax.set_ylim(-2,2)
    if i == ndim - 1:
        ax.set_xlabel("Iterations")
    else:
        ax.set_xticks([])
    ax.set_ylabel(r"$x_{" + str(i) + "}$")

    ax.set_xlim(0, nsteps)

plt.tight_layout()
plt.show()

# %%
# corner plot
corner(iid, labels=[r"$\alpha$", r"$\beta$"], truths=[alpha_mean, beta_mean])
plt.show()
