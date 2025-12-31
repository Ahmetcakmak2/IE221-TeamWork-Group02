import os
import numpy as np
import matplotlib.pyplot as plt

# Number of samples
n = 10000

# True mean of Uniform(0,1)
mu = 0.5

# Generate samples
x = np.random.uniform(0, 1, n)

# Cumulative mean
cumulative_mean = np.cumsum(x) / np.arange(1, n + 1)

# Output directory
out_dir = os.path.join("results", "figures")
os.makedirs(out_dir, exist_ok=True)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(cumulative_mean, label="Cumulative Mean")
plt.axhline(mu, linestyle="--", label="True Mean = 0.5")
plt.xlabel("n")
plt.ylabel("Cumulative Mean")
plt.title("SLLN Simulation for Uniform(0,1)")
plt.legend()
plt.tight_layout()

# Save figure
out_path = os.path.join(out_dir, "slln_convergence.png")
plt.savefig(out_path)
plt.close()

print("Bitti → results/figures/slln_convergence.png")
