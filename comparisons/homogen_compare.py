#!/usr/bin/env python3
"""
Compare Hoeffding (Thm 1 & 2), Tight Chernoff, and Bentkus bounds for sums of bounded 
random variables where all variables are in [0, 1] (homogeneous intervals).

In this case:
- Hoeffding Theorem 1 (KL-based) applies directly
- Hoeffding Theorem 2 (sum of squares) also applies
- Tight Chernoff (sharp Chernoff) should match Hoeffding Theorem 1 as a sanity check
- Bentkus Thm 1.2 bound compares to Binomial(n, p)
- Bentkus Cor 1.4 bound uses RMS scale with symmetric Bernoulli comparison

This provides a sanity check that Tight Chernoff agrees with the KL Hoeffding bound
when all intervals are equal to [0, 1].
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path to import becitbam
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from becitbam import hoeffding_thm1, hoeffding_thm2, sharp_chernoff, bentkus, bentkus_binomial


def compute_bounds(s_values, mu, n):
    """
    Compute all bounds for P(S > s) with homogeneous intervals [0, 1].
    
    Parameters
    ----------
    s_values : array-like
        Values of s to evaluate
    mu : float
        Mean of the sum S
    n : int
        Number of variables (each X_i in [0, 1])
    
    Returns
    -------
    hoeffding1_bounds : numpy.ndarray
        Hoeffding Theorem 1 (KL) bound on P(S > s)
    hoeffding2_bounds : numpy.ndarray
        Hoeffding Theorem 2 (sum of squares) bound on P(S > s)
    tight_chernoff_bounds : numpy.ndarray
        Tight Chernoff (sharp Chernoff) bound on P(S > s)
    bentkus_thm12_bounds : numpy.ndarray
        Bentkus Thm 1.2 bound on P(S > s)
    bentkus_cor14_bounds : numpy.ndarray
        Bentkus Cor 1.4 bound on P(S > s) (using RMS scale)
    """
    # All intervals are [0, 1]
    a = np.ones(n)
    
    hoeffding1_bounds = []
    hoeffding2_bounds = []
    tight_chernoff_bounds = []
    bentkus_thm12_bounds = []
    bentkus_cor14_bounds = []
    
    for s in s_values:
        # Hoeffding Theorem 1 (KL-based, returns log probability)
        log_hoeffding1 = hoeffding_thm1(s, n, mu)
        hoeffding1_bounds.append(np.exp(log_hoeffding1))
        
        # Hoeffding Theorem 2 (sum of squares, returns log probability)
        log_hoeffding2 = hoeffding_thm2(s, mu, a)
        hoeffding2_bounds.append(np.exp(log_hoeffding2))
        
        # Tight Chernoff bound (returns log probability)
        log_tight_chernoff = sharp_chernoff(s, mu, a)
        tight_chernoff_bounds.append(np.exp(log_tight_chernoff))
        
        # Bentkus Thm 1.2 bound (returns probability directly)
        bentkus_thm12_prob = bentkus(s, mu, a)
        bentkus_thm12_bounds.append(bentkus_thm12_prob)
        
        # Bentkus Cor 1.4 bound (returns probability directly)
        bentkus_cor14_prob = bentkus_binomial(s, mu, a)
        bentkus_cor14_bounds.append(bentkus_cor14_prob)
    
    return (np.array(hoeffding1_bounds), np.array(hoeffding2_bounds), 
            np.array(tight_chernoff_bounds), np.array(bentkus_thm12_bounds),
            np.array(bentkus_cor14_bounds))


def main(n=100):
    """
    Main function to generate comparison plots.
    
    Parameters
    ----------
    n : int
        Number of variables (default: 100)
    """
    print(f"Homogeneous intervals comparison with n={n} variables in [0, 1]")
    print(f"Sum of all intervals: {n}")
    
    # Values of s to evaluate (range from 80 to 100 for n=100)
    s_min = 0.8 * n
    s_max = 1.0 * n
    s_values = np.linspace(s_min, s_max, 100)
    
    # Values of mu to consider (scaled by n)
    mu_fractions = [0.8, 0.9, 0.95]
    mu_values = [frac * n for frac in mu_fractions]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and styles for different mu values
    colors = ['blue', 'green', 'red']
    
    for mu, frac, color in zip(mu_values, mu_fractions, colors):
        print(f"Computing bounds for mu={mu} (fraction={frac})...")
        hoeffding1, hoeffding2, tight_chernoff, bentkus_thm12, bentkus_cor14 = compute_bounds(s_values, mu, n)
        
        # Plot Hoeffding Theorem 1 (KL) bound (dashed line)
        ax.plot(s_values / n, hoeffding1, '--', color=color, 
                label=f'Hoeffding KL (μ={frac})', linewidth=1.5)
        
        # Plot Hoeffding Theorem 2 (sum of squares) bound (dash-dot line)
        ax.plot(s_values / n, hoeffding2, '-.', color=color, 
                label=f'Hoeffding Σa² (μ={frac})', linewidth=1.5)
        
        # Plot Tight Chernoff bound (solid line)
        ax.plot(s_values / n, tight_chernoff, '-', color=color,
                label=f'Tight Chernoff (μ={frac})', linewidth=1.5)
        
        # Plot Bentkus Thm 1.2 bound (dotted line)
        ax.plot(s_values / n, bentkus_thm12, ':', color=color,
                label=f'Bentkus Thm 1.2 (μ={frac})', linewidth=1.5)
        
        # Plot Bentkus Cor 1.4 bound (markers)
        ax.plot(s_values / n, bentkus_cor14, 'x', color=color,
                label=f'Bentkus Cor 1.4 (μ={frac})', markersize=4, markevery=5)
        
        # Sanity check: Tight Chernoff should match Hoeffding KL
        max_diff = np.max(np.abs(hoeffding1 - tight_chernoff))
        print(f"  Max |Hoeffding KL - Tight Chernoff| = {max_diff:.2e}")
    
    ax.set_xlabel('s/n (normalized threshold)', fontsize=12)
    ax.set_ylabel('P(S > s)', fontsize=12)
    ax.set_title(f'Comparison of Bounds with Homogeneous Intervals\n(n={n}, all X_i in [0, 1])', fontsize=14)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 1.0)
    
    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'homogen_{n}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare bounds with homogeneous intervals')
    parser.add_argument('--n', type=int, default=100, help='Number of variables (default: 100)')
    
    args = parser.parse_args()
    main(n=args.n)
