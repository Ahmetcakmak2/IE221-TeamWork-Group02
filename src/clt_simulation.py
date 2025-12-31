import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# U[0,1] parameters
mu = 0.5
sigma = np.sqrt(1/12)

# CLT settings
n_values = [2, 5, 10, 30, 50]
m = 1000

os.makedirs("results/figures", exist_ok=True)

for n in n_values:
    # m experiments, each summing n uniforms
    samples = np.random.uniform(0, 1, (m, n))
    sums = samples.sum(axis=1)

    # Standardization
    Z = (sums - n * mu) / (sigma * np.sqrt(n))

    # Histogram
    plt.figure()
    plt.hist(Z, bins=30, density=True)
    x = np.linspace(-4, 4, 400)
    plt.plot(x, stats.norm.pdf(x))
    plt.xlabel("Z")
    plt.ylabel("Density")
    plt.title(f"CLT Histogram (n={n})")
    plt.savefig(f"results/figures/clt_histogram_n{n}.png")
    plt.close()

    # Q-Q plot
    plt.figure()
    stats.probplot(Z, dist="norm", plot=plt)
    plt.title(f"CLT Q-Q Plot (n={n})")
    plt.savefig(f"results/figures/clt_qqplot_n{n}.png")
    plt.close()
