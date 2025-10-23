# -*- coding: utf-8 -*-

import numpy as np
import jax
from chebyshev_utils import get_coeffs, sample_from_coeff, chebval_ab_for_one_x
import matplotlib.pyplot as plt
import scipy

if __name__ == '__main__':
    
    # Setup
    a, b = -8, 8
    num_samples = 10**6
    key = jax.random.PRNGKey(355353)
    
    def f(x):
        return np.exp(-x**2/2) * (1 + np.square(np.sin(3*x))) * (1 + np.square(np.cos(5*x)))
    
    # Compute normalizing constant
    normalizing_constant = scipy.integrate.quad(
        f, a, b, limit=500, epsabs=1.49e-011, epsrel=1.49e-08, maxp1=500, limlst=500
    )
    
    # Compute approximations and errors
    N_values = np.arange(25, 205, 5)
    coeff_N = dict()
    errors_N_values = []
    
    x_lin = np.linspace(a, b, 20000)
    y_true = f(x_lin)
    
    for N in N_values:
        coeff = get_coeffs(f, a, b, N)
        approx_values = chebval_ab_for_one_x(x_lin, coeff, a, b)
        error = np.max(np.abs(approx_values - y_true))
        errors_N_values.append(error)
        coeff_N[N] = coeff
    
    # Generate samples from approximation
    N_best = N_values[-1]
    N_to_use = coeff_N[N_best]
    samples = sample_from_coeff(N_to_use, key, a, b, num_samples)
    
    # Plot 1: Function Approximation Comparison (three values of N)
    N_comparison = [25, 125, 200]
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (ax, N) in enumerate(zip(axes, N_comparison)):
        if N in coeff_N:
            approx = chebval_ab_for_one_x(x_lin, coeff_N[N], a, b)
            error = approx - y_true
            
            ax.plot(x_lin, y_true, 'k-', linewidth=2, label='True function')
            ax.plot(x_lin, approx, '--', linewidth=2, label=f'Approximation N={N}')
            
            # Plot error (scaled for visibility)
            ax.plot(x_lin, error , linewidth=1.5, 
                   label='Error')
            
            ax.set_xlabel('x')
            ax.set_ylabel('Value')
            ax.set_title(f'N = {N}, Max Error: {np.max(np.abs(error)):.2e}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(a, b)
    
    plt.tight_layout()
    plt.suptitle('Function Approximation Comparison', fontsize=16, y=1.02)
    plt.savefig('Unnormnalized_approx.pdf', dpi = 900, bbox_inches = 'tight')
    
    # Plot 2: Error Rate (simple logarithmic plot)
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(N_values, errors_N_values, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Approximation Order N')
    ax.set_ylabel('Maximum Absolute Error')
    ax.set_title('Convergence of the Approximation Error')
    ax.grid(True, alpha=0.3)
    plt.savefig('error_rate.pdf', dpi = 900, bbox_inches = 'tight')
    # Plot 3: Normalized Function vs Samples (simple)
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    y_normalized = y_true / normalizing_constant[0]
    ax.plot(x_lin, y_normalized, 'r-', linewidth=2, 
            label=f'Normalized PDF')
    
    ax.hist(samples, density=True, bins=100, alpha=0.6, 
            color='lightblue', edgecolor='blue', linewidth=0.5,
            label=f'Samples (n={num_samples:.0e})')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('True PDF vs Generated Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(a, b)
    
    plt.savefig('normalized_vs_samples.pdf', dpi = 900, bbox_inches = 'tight')
    plt.show()
