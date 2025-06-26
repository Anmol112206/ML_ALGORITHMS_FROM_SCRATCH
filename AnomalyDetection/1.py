import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


data = pd.read_csv('data/normal_data.csv')  
X = data.values  # Convert to NumPy array


def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    sigma = np.var(X, axis=0)
    return mu, sigma

mu, sigma = estimate_gaussian(X)


def gaussian_probability(X, mu, sigma):
    p = np.prod(1 / (np.sqrt(2 * np.pi * sigma)) *  np.exp(-(X - mu) ** 2 / (2 * sigma)), axis=1)
    return p

p = gaussian_probability(X, mu, sigma)


#Choose threshold ε (using CV set or trial-and-error)
epsilon = np.percentile(p, 1)  #e.g., bottom 1% as anomalies


anomalies = p < epsilon


if X.shape[1] == 2:
    plt.figure(figsize=(8,6))
    plt.scatter(X[:, 0], X[:, 1], label='Normal Data')
    plt.scatter(X[anomalies, 0], X[anomalies, 1], color='r', marker='x', label='Anomalies')
    plt.title("Anomaly Detection")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()    
else:
    print(f"Total anomalies found: {np.sum(anomalies)}")

for i in range(X.shape[1]):
    plt.figure(figsize=(6, 4))
    plt.hist(X[:, i], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of Feature {i + 1}')
    plt.xlabel(f'Feature {i + 1} values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()