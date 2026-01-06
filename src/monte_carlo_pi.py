import os
import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo_simulation(n_points: int = 100000):
    """
    Runs a Monte Carlo simulation to estimate the value of Pi.

    This function generates random points within a unit square [0,1]x[0,1]
    and determines the ratio of points falling inside the unit quarter-circle.
    
    The estimation is based on the formula: Pi approx = 4 * (Inside / Total)

    Args:
        n_points (int): The number of random points to generate. 
                        Default is 100,000.

    Returns:
        None: The function saves the resulting plot to 'results/figures/'.
    """
    
    # 1. Output directory setup (Create folder if not exists)
    out_dir = os.path.join("..", "results", "figures") # '..' goes up one level from src
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting simulation with {n_points} points...")

    # 2. Generate random points using NumPy (Vectorized operation for speed)
    # Uniform distribution between 0 and 1
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)

    # 3. Check if points are inside the quarter circle (x^2 + y^2 <= 1)
    inside_mask = (x**2 + y**2) <= 1
    
    # Calculate cumulative sum of points inside the circle
    inside_cumsum = np.cumsum(inside_mask)
    
    # 4. Calculate Pi estimation at each step
    # Formula: Pi = 4 * (Cumulative Inside / Current Sample Size)
    sample_sizes = np.arange(1, n_points + 1)
    pi_estimates = 4 * inside_cumsum / sample_sizes

    # Final estimate
    final_pi = pi_estimates[-1]
    print(f"Final Pi Estimation: {final_pi:.5f}")

    # 5. Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(pi_estimates, label='Monte Carlo Estimation', color='blue')
    plt.axhline(np.pi, color='red', linestyle='--', label='True Pi (3.1415...)')
    
    plt.xlabel('Number of Iterations (n)')
    plt.ylabel('Estimated Value of Pi')
    plt.title(f'Monte Carlo Estimation of Pi (n={n_points})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 6. Save the figure
    output_path = os.path.join(out_dir, "pi_estimation.png")
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # This block executes only when the script is run directly
    run_monte_carlo_simulation(n_points=100000)
