#!/usr/bin/env python3
"""
Compare Hoeffding and Becitbam bounds for sums of bounded random variables
where the bounds come from a Dirichlet (n-simplex) distribution.

This script generates a random vector `a` from the n-simplex, considers
n independent variables where each variable X_i is bounded in [0, a_i],
assumes the overall mean is known to be mu, and evaluates bounds on P(S > s)
for the sum S = sum_i X_i.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path to import becitbam
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from becitbam import hoeffding_thm2, sharp_chernoff


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
    Compute Hoeffding and Becitbam bounds for P(S > s).
    
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
    """
    hoeffding_bounds = []
    becitbam_bounds = []
    
    for s in s_values:
        # Hoeffding bound (returns log probability)
        log_hoeffding = hoeffding_thm2(s, mu, a)
        hoeffding_bounds.append(np.exp(log_hoeffding))
        
        # Becitbam sharp Chernoff bound (returns log probability)
        log_becitbam = sharp_chernoff(s, mu, a)
        becitbam_bounds.append(np.exp(log_becitbam))
    
    return np.array(hoeffding_bounds), np.array(becitbam_bounds)


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
    print(f"Generated simplex vector with n={n}, sum(a)={A:.6f}")
    
    # Values of s to evaluate (range from 0.8 to 1.0)
    s_values = np.linspace(0.8, 1.0, 100)
    
    # Values of mu to consider
    mu_values = [0.8, 0.9, 0.95]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and styles for different mu values
    colors = ['blue', 'green', 'red']
    
    for mu, color in zip(mu_values, colors):
        print(f"Computing bounds for mu={mu}...")
        hoeffding_bounds, becitbam_bounds = compute_bounds(s_values, mu, a)
        
        # Plot Hoeffding bound (dashed line)
        ax.plot(s_values, hoeffding_bounds, '--', color=color, 
                label=f'Hoeffding (μ={mu})', linewidth=1.5)
        
        # Plot Becitbam bound (solid line)
        ax.plot(s_values, becitbam_bounds, '-', color=color,
                label=f'Becitbam (μ={mu})', linewidth=1.5)
    
    ax.set_xlabel('s', fontsize=12)
    ax.set_ylabel('P(S > s)', fontsize=12)
    ax.set_title(f'Comparison of Hoeffding and Becitbam Bounds\n(n={n}, simplex-distributed intervals)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
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
    
    parser = argparse.ArgumentParser(description='Compare Hoeffding and Becitbam bounds')
    parser.add_argument('--n', type=int, default=100, help='Dimension of the simplex (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    main(n=args.n, seed=args.seed)
