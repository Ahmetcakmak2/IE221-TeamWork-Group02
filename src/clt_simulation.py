import numpy as np
import matplotlib.pyplot as plt
import os

def run_clt_simulation(n_experiments: int = 5000, sample_size: int = 30):
    """
    Demonstrates the Central Limit Theorem (CLT) simulation.

    This function draws multiple samples from a non-normal distribution (Uniform),
    calculates the mean of each sample, and plots the histogram of these sample means.
    According to CLT, this histogram should resemble a Normal Distribution (Bell Curve).

    Args:
        n_experiments (int): Number of times to repeat the sampling process.
        sample_size (int): The size of each sample group.
                           (Usually n > 30 is sufficient for CLT).

    Returns:
        None: Saves the histogram plot to 'results/figures/clt_plot.png'.
    """
    
    # 1. Setup output directory
    out_dir = os.path.join("..", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running CLT Simulation ({n_experiments} experiments, sample size={sample_size})...")

    sample_means = []

    # 2. Run Experiments
    for _ in range(n_experiments):
        # Draw random samples from a Uniform Distribution [0, 1]
        # This is NOT a normal distribution, which makes it perfect for testing CLT.
        sample = np.random.uniform(0, 1, sample_size)
        
        # Calculate the mean of this sample group
        mean_val = np.mean(sample)
        sample_means.append(mean_val)

    # 3. Plotting Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(sample_means, bins=50, density=True, alpha=0.7, color='green', edgecolor='black', label='Sample Means')
    
    plt.title(f'Central Limit Theorem (CLT) Verification\n(Distribution of Sample Means, n={sample_size})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 4. Save Figure
    output_path = os.path.join(out_dir, "clt_plot.png")
    plt.savefig(output_path)
    print(f"CLT plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    run_clt_simulation()
