#!/usr/bin/env python3
"""
Compare Hoeffding (Thm 1 & 2), Becitbam, and Bentkus bounds for sums of bounded 
random variables where all variables are in [0, 1] (homogeneous intervals).

In this case:
- Hoeffding Theorem 1 (KL-based) applies directly
- Hoeffding Theorem 2 (sum of squares) also applies
- Becitbam (sharp Chernoff) should match Hoeffding Theorem 1 as a sanity check
- Bentkus bound compares to Binomial(n, p)

This provides a sanity check that Becitbam agrees with the KL Hoeffding bound
when all intervals are equal to [0, 1].
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path to import becitbam
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from becitbam import hoeffding_thm1, hoeffding_thm2, sharp_chernoff, bentkus


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
    becitbam_bounds : numpy.ndarray
        Becitbam (sharp Chernoff) bound on P(S > s)
    bentkus_bounds : numpy.ndarray
        Bentkus bound on P(S > s)
    """
    # All intervals are [0, 1]
    a = np.ones(n)
    
    hoeffding1_bounds = []
    hoeffding2_bounds = []
    becitbam_bounds = []
    bentkus_bounds = []
    
    for s in s_values:
        # Hoeffding Theorem 1 (KL-based, returns log probability)
        log_hoeffding1 = hoeffding_thm1(s, n, mu)
        hoeffding1_bounds.append(np.exp(log_hoeffding1))
        
        # Hoeffding Theorem 2 (sum of squares, returns log probability)
        log_hoeffding2 = hoeffding_thm2(s, mu, a)
        hoeffding2_bounds.append(np.exp(log_hoeffding2))
        
        # Becitbam sharp Chernoff bound (returns log probability)
        log_becitbam = sharp_chernoff(s, mu, a)
        becitbam_bounds.append(np.exp(log_becitbam))
        
        # Bentkus bound (returns probability directly)
        bentkus_prob = bentkus(s, mu, a)
        bentkus_bounds.append(bentkus_prob)
    
    return (np.array(hoeffding1_bounds), np.array(hoeffding2_bounds), 
            np.array(becitbam_bounds), np.array(bentkus_bounds))


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
        hoeffding1, hoeffding2, becitbam, bentkus_bounds = compute_bounds(s_values, mu, n)
        
        # Plot Hoeffding Theorem 1 (KL) bound (dashed line)
        ax.plot(s_values / n, hoeffding1, '--', color=color, 
                label=f'Hoeffding KL (μ={frac})', linewidth=1.5)
        
        # Plot Hoeffding Theorem 2 (sum of squares) bound (dash-dot line)
        ax.plot(s_values / n, hoeffding2, '-.', color=color, 
                label=f'Hoeffding Σa² (μ={frac})', linewidth=1.5)
        
        # Plot Becitbam bound (solid line)
        ax.plot(s_values / n, becitbam, '-', color=color,
                label=f'Becitbam (μ={frac})', linewidth=1.5)
        
        # Plot Bentkus bound (dotted line)
        ax.plot(s_values / n, bentkus_bounds, ':', color=color,
                label=f'Bentkus (μ={frac})', linewidth=1.5)
        
        # Sanity check: Becitbam should match Hoeffding KL
        max_diff = np.max(np.abs(hoeffding1 - becitbam))
        print(f"  Max |Hoeffding KL - Becitbam| = {max_diff:.2e}")
    
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
