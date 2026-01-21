#!/usr/bin/env python3
"""
Compare Hoeffding, Becitbam, and Bentkus bounds for sums of bounded random variables
where the bounds come from a Dirichlet (n-simplex) distribution.

This script generates a random vector `a` from the n-simplex, considers
n independent variables where each variable X_i is bounded in [0, a_i],
assumes the overall mean is known to be mu, and evaluates bounds on P(S > s)
for the sum S = sum_i X_i.

The Bentkus bound rescales all intervals by the maximum interval width so that
max(a_i) = 1, then compares to a Binomial(n, p) distribution where p = mu_rescaled/n.
To avoid log-concave interpolation, Bentkus bounds are only plotted at s values
where the rescaled s is an integer.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import sys
import os

# Add the parent directory to path to import becitbam
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from becitbam import hoeffding_thm2, sharp_chernoff


def bentkus_bound(s, mu, a):
    """
    Compute Bentkus bound for P(S >= s) by rescaling to maximum interval width = 1.
    
    Based on Bentkus (2004) Theorem 1.2: For independent random variables X_i in [0, 1]
    with total mean mu, we have P(S >= s) <= e * P(Binomial(n, p) >= s) where p = mu/n.
    
    This function rescales the problem so that max(a_i) = 1, then applies the bound.
    For the bound to apply without log-concave interpolation, s (after rescaling)
    should be an integer.
    
    Parameters
    ----------
    s : float
        Threshold value
    mu : float
        Mean of the sum S
    a : numpy.ndarray
        Upper bounds for each variable (X_i in [0, a_i])
    
    Returns
    -------
    float
        Bentkus bound on P(S >= s) (probability, not log probability)
    """
    n = len(a)
    b = np.max(a)  # maximum interval width
    
    # Rescale: after rescaling, all variables are in [0, a_i/b] which is in [0, 1]
    # The rescaled threshold and mean
    s_rescaled = s / b
    mu_rescaled = mu / b
    
    # Mean probability per variable for the comparison Binomial
    p = mu_rescaled / n
    
    # Ensure p is in valid range [0, 1]
    p = np.clip(p, 0.0, 1.0)
    
    # For integer s_rescaled: P(Binomial(n, p) >= s_rescaled)
    # Round up to nearest integer to be conservative
    s_int = int(np.ceil(s_rescaled))
    
    if s_int > n:
        return 0.0
    if s_int <= 0:
        return 1.0
    
    # P(Binomial(n, p) >= s_int) = survival function at s_int - 1
    binomial_tail = binom.sf(s_int - 1, n, p)
    
    # Bentkus constant is e ≈ 2.72
    bentkus_prob = np.e * binomial_tail
    
    # Probability cannot exceed 1
    return min(bentkus_prob, 1.0)


def generate_simplex_vector(n, seed=42):
    """
    Generate a random vector from the n-simplex using Dirichlet(1,1,...,1) distribution.
    The resulting vector has components that sum to 1.
    
    Parameters
    ----------
    n : int
        Dimension of the simplex
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    a : numpy.ndarray
        Random vector from the n-simplex (sums to 1)
    """
    rng = np.random.default_rng(seed)
    # Dirichlet(1,1,...,1) is uniform on the simplex
    a = rng.dirichlet(np.ones(n))
    return a


def compute_bounds(s_values, mu, a):
    """
    Compute Hoeffding, Becitbam, and Bentkus bounds for P(S > s).
    
    Parameters
    ----------
    s_values : array-like
        Values of s to evaluate
    mu : float
        Mean of the sum S
    a : numpy.ndarray
        Upper bounds for each variable (X_i in [0, a_i])
    
    Returns
    -------
    hoeffding_bounds : numpy.ndarray
        Hoeffding bound on P(S > s) for each s
    becitbam_bounds : numpy.ndarray
        Becitbam (sharp Chernoff) bound on P(S > s) for each s
    bentkus_bounds : numpy.ndarray
        Bentkus bound on P(S > s) for each s (rescaled to max interval = 1)
    """
    hoeffding_bounds = []
    becitbam_bounds = []
    bentkus_bounds = []
    
    for s in s_values:
        # Hoeffding bound (returns log probability)
        log_hoeffding = hoeffding_thm2(s, mu, a)
        hoeffding_bounds.append(np.exp(log_hoeffding))
        
        # Becitbam sharp Chernoff bound (returns log probability)
        log_becitbam = sharp_chernoff(s, mu, a)
        becitbam_bounds.append(np.exp(log_becitbam))
        
        # Bentkus bound (returns probability directly)
        bentkus_prob = bentkus_bound(s, mu, a)
        bentkus_bounds.append(bentkus_prob)
    
    return np.array(hoeffding_bounds), np.array(becitbam_bounds), np.array(bentkus_bounds)


def main(n=100, seed=42):
    """
    Main function to generate comparison plots.
    
    Parameters
    ----------
    n : int
        Dimension of the simplex (default: 100)
    seed : int
        Random seed for reproducibility (default: 42)
    """
    # Generate random vector from the n-simplex
    a = generate_simplex_vector(n, seed=seed)
    
    # The sum of a is 1 (since it's from the simplex)
    A = np.sum(a)  # Should be 1.0
    b = np.max(a)  # Maximum interval width
    print(f"Generated simplex vector with n={n}, sum(a)={A:.6f}, max(a)={b:.6f}")
    
    # For Bentkus bound, we use integer s values after rescaling by 1/max(a)
    # s_rescaled = s / b should be integer, so s = k * b for integer k
    # We want s in range [0.8, 1.0], so k ranges from ceil(0.8/b) to floor(1.0/b)
    k_min = int(np.ceil(0.8 / b))
    k_max = int(np.floor(1.0 / b))
    bentkus_s_values = np.array([k * b for k in range(k_min, k_max + 1)])
    print(f"Bentkus s values (rescaled integers): {len(bentkus_s_values)} points from k={k_min} to k={k_max}")
    
    # Values of s to evaluate for Hoeffding/Becitbam (continuous)
    s_values = np.linspace(0.8, 1.0, 100)
    
    # Values of mu to consider
    mu_values = [0.8, 0.9, 0.95]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and styles for different mu values
    colors = ['blue', 'green', 'red']
    
    for mu, color in zip(mu_values, colors):
        print(f"Computing bounds for mu={mu}...")
        hoeffding_bounds, becitbam_bounds, _ = compute_bounds(s_values, mu, a)
        
        # Compute Bentkus bounds only at integer-rescaled s values
        bentkus_bounds = np.array([bentkus_bound(s, mu, a) for s in bentkus_s_values])
        
        # Plot Hoeffding bound (dashed line)
        ax.plot(s_values, hoeffding_bounds, '--', color=color, 
                label=f'Hoeffding (μ={mu})', linewidth=1.5)
        
        # Plot Becitbam bound (solid line)
        ax.plot(s_values, becitbam_bounds, '-', color=color,
                label=f'Becitbam (μ={mu})', linewidth=1.5)
        
        # Plot Bentkus bound (dotted line with markers)
        ax.plot(bentkus_s_values, bentkus_bounds, ':', color=color,
                label=f'Bentkus (μ={mu})', linewidth=1.5, marker='o', markersize=3)
    
    ax.set_xlabel('s', fontsize=12)
    ax.set_ylabel('P(S > s)', fontsize=12)
    ax.set_title(f'Comparison of Hoeffding, Becitbam, and Bentkus Bounds\n(n={n}, simplex-distributed intervals)', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 1.0)
    
    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'dirichlet_{n}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Hoeffding, Becitbam, and Bentkus bounds')
    parser.add_argument('--n', type=int, default=100, help='Dimension of the simplex (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    main(n=args.n, seed=args.seed)
