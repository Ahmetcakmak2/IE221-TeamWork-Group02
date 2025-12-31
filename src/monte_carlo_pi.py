import os
import numpy as np
import matplotlib.pyplot as plt

# Number of random points
n = 100000

# Output directory
out_dir = os.path.join("results", "figures")
os.makedirs(out_dir, exist_ok=True)

# Generate random points in [0,1]x[0,1]
x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)

# Check if inside quarter circle
inside = (x**2 + y**2) <= 1

# Cumulative count
inside_cum = np.cumsum(inside)

# Monte Carlo estimation of pi
pi_est = 4 * inside_cum / np.arange(1, n + 1)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(pi_est, label="π estimate")
plt.axhline(np.pi, linestyle="--", label="True π")
plt.xlabel("n")
plt.ylabel("π estimate")
plt.title("Monte Carlo Estimation of π")
plt.legend()
plt.tight_layout()

# Save figure
out_path = os.path.join(out_dir, "pi_estimation.png")
plt.savefig(out_path)
plt.close()

print("Bitti → results/figures/pi_estimation.png")
