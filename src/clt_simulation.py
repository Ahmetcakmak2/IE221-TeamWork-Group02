import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mu = 0.5
sigma = np.sqrt(1/12)

n_values = [2, 5, 10, 30, 50]
m = 1000

os.makedirs("results/figures", exist_ok=True)

for n in n_values:
    data = np.random.uniform(0, 1, (m, n))
    sums = data.sum(axis=1)
    Z = (sums - n * mu) / (sigma * np.sqrt(n))

    plt.figure()
    plt.hist(Z, bins=30, density=True)
    x = np.linspace(-4, 4, 400)
    plt.plot(x, stats.norm.pdf(x))
    plt.savefig(f"results/figures/clt_histogram_n{n}.png")
    plt.close()

    plt.figure()
    stats.probplot(Z, dist="norm", plot=plt)
    plt.savefig(f"results/figures/clt_qqplot_n{n}.png")
    plt.close()
