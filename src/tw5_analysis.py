import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Results klasörü yoksa oluştur
if not os.path.exists('../results'):
    os.makedirs('../results')

def get_theoretical_moments(dist_name):
    # Dağılımların teorik değerleri (Ödevdeki tabloya göre)
    if dist_name == "Uniform":
        return 0.5, 1/12
    elif dist_name == "Exponential":
        return 1.0, 1.0
    elif dist_name == "Pareto_3":
        return 1.5, 0.75
    elif dist_name == "Pareto_1.5":
        return 3.0, np.inf  # Varyans sonsuz
    elif dist_name == "Cauchy":
        return None, None   # Tanımsız
    return None, None

def generate_data(dist_name, size):
    # Veri üretme fonksiyonları
    if dist_name == "Uniform":
        return np.random.uniform(0, 1, size)
    elif dist_name == "Exponential":
        return np.random.exponential(1, size)
    elif dist_name == "Pareto_3":
        return (np.random.pareto(3.0, size) + 1)
    elif dist_name == "Pareto_1.5":
        return (np.random.pareto(1.5, size) + 1)
    elif dist_name == "Cauchy":
        return np.random.standard_cauchy(size)

def analyze_slln(dist_name, n=10000):
    # SLLN Analizi (Kümülatif Ortalama)
    data = generate_data(dist_name, n)
    cumulative_means = np.cumsum(data) / np.arange(1, n + 1)
    mu, _ = get_theoretical_moments(dist_name)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_means, label='Sample Mean')
    if mu is not None:
        plt.axhline(y=mu, color='r', linestyle='--', label=f'Theoretical Mean ({mu})')
    
    plt.title(f'SLLN Analysis: {dist_name}')
    plt.xlabel('n')
    plt.ylabel('Cumulative Mean')
    plt.legend()
    plt.savefig(f'../results/{dist_name}_SLLN.png')
    plt.close()

def analyze_clt(dist_name, n_values=[2, 5, 10, 30, 50, 100], m=1000):
    # CLT Analizi (Histogram ve Q-Q Plot)
    mu, sigma2 = get_theoretical_moments(dist_name)
    sigma = np.sqrt(sigma2) if sigma2 is not None else None
    
    for n in n_values:
        sample_sums = []
        for _ in range(m):
            sample = generate_data(dist_name, n)
            sample_sums.append(np.sum(sample))
        
        sample_sums = np.array(sample_sums)
        
        # Standardizasyon (Mümkünse)
        if mu is not None and sigma is not None and not np.isinf(sigma):
            data_to_plot = (sample_sums - n * mu) / (sigma * np.sqrt(n))
        else:
            data_to_plot = sample_sums

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.hist(data_to_plot, bins=30, density=True, alpha=0.6, color='g')
        ax1.set_title(f'{dist_name} Histogram (n={n})')
        stats.probplot(data_to_plot, dist="norm", plot=ax2)
        ax2.set_title(f'{dist_name} Q-Q Plot (n={n})')
        
        plt.tight_layout()
        plt.savefig(f'../results/{dist_name}_CLT_n{n}.png')
        plt.close()

# --- ÇALIŞTIRMA KISMI ---
distributions = ["Uniform", "Exponential", "Pareto_3", "Pareto_1.5", "Cauchy"]
print("Analiz başlıyor, lütfen bekleyin...")
for dist in distributions:
    analyze_slln(dist)
    analyze_clt(dist)
    print(f"{dist} tamamlandı.")
print("Bitti! Grafikler results klasöründe.")