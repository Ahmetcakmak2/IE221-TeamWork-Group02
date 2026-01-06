import numpy as np
import matplotlib.pyplot as plt
import os

def run_slln_simulation(n_samples: int = 2000):
    """
    Demonstrates the Strong Law of Large Numbers (SLLN) simulation.
    
    This function generates a sequence of random variables (coin flips or dice rolls)
    and plots the cumulative average. As the number of samples (n) increases,
    the sample mean converges to the true expected value.

    Args:
        n_samples (int): The total number of random experiments to perform.
                         Default is 2000.
    
    Returns:
        None: Saves the convergence plot to 'results/figures/slln_plot.png'.
    """
    
    # 1. Setup output directory
    out_dir = os.path.join("..", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Running SLLN Simulation with {n_samples} samples...")

    # 2. Generate Data: Simulating a fair die roll (1 to 6)
    # Expected Value (True Mean) = 3.5
    data = np.random.randint(1, 7, n_samples)
    true_mean = 3.5

    # 3. Calculate Cumulative Averages
    # cumsum() -> [x1, x1+x2, x1+x2+x3 ...]
    cumulative_sum = np.cumsum(data)
    sample_sizes = np.arange(1, n_samples + 1)
    cumulative_averages = cumulative_sum / sample_sizes

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_averages, label='Sample Mean', color='blue')
    plt.axhline(y=true_mean, color='red', linestyle='--', label=f'True Mean ({true_mean})')
    
    plt.title('Strong Law of Large Numbers (SLLN) Verification')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Cumulative Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Save Figure
    output_path = os.path.join(out_dir, "slln_plot.png")
    plt.savefig(output_path)
    print(f"SLLN plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    run_slln_simulation()
