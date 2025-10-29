# -*- coding: utf-8 -*-
from src.dataloader.generate_sup_ig_nig_5params import slice_sample_sup_ig_nig_trawl
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.KL_divergence import convert_3_to_4_param_nig
from functools import partial
import jax.numpy as jnp
import jax

@jax.jit
def corr_sup_ig_envelope(h, params):
    gamma, eta = params
    return jnp.exp(eta * (1 - jnp.sqrt(1 + 2*h/gamma**2)))

def estimate_bias_from_MAP(MAP, observed_trawl, num_replicated_for_bias_estimation, nlags, key):
    
    vec_simulator = jax.jit(
        jax.vmap(slice_sample_sup_ig_nig_trawl, in_axes=(None, None, None, None, 0)),
        static_argnums=(0, 1)
    )
    
    acf_theta = MAP[:2]
    marginal_theta = convert_3_to_4_param_nig(MAP[2:])
    
    # Batch processing to avoid memory issues
    num_batches = 4
    batch_size = num_replicated_for_bias_estimation // num_batches
    all_ACFs = []
    
    for batch_idx in range(num_batches):
        # Generate keys for this batch
        key, subkey = jax.random.split(key)
        batch_keys = jax.random.split(subkey, batch_size)
        
        # Simulate batch
        simulated_trawls_batch, _ = vec_simulator(observed_trawl.shape[-1], 1.0, acf_theta, marginal_theta, 
                                                  batch_keys)
        
        # Compute ACFs for this batch
        batch_ACFs = jnp.array([compute_empirical_acf(i, adjusted=True, nlags=nlags) 
                                for i in simulated_trawls_batch])
        all_ACFs.append(batch_ACFs)
        
        # Clear batch from memory
        del simulated_trawls_batch
    
    # Concatenate all ACFs
    all_ACFs = jnp.concatenate(all_ACFs, axis=0)  # Shape: (num_replicated, nlags+1)
    
    # Compute theoretical ACF
    theoretical_ACF = corr_sup_ig_envelope(jnp.arange(0, nlags+1, 1), MAP[:2])
    
    # Compute differences from theoretical
    differences = all_ACFs - theoretical_ACF  # Shape: (num_replicated, nlags+1)
    
    # Compute statistics
    mean_bias = jnp.mean(differences, axis=0)
    lower_quantile = jnp.percentile(differences, 2.5, axis=0)
    upper_quantile = jnp.percentile(differences, 97.5, axis=0)
    
    return mean_bias, lower_quantile, upper_quantile

def estimate_bias_from_posterior_samples_batched(posterior_samples, observed_trawl, nlags, key, batch_size=100):
    n_samples = posterior_samples.shape[0]
    all_diff = []
    
    for i in range(0, n_samples, batch_size):
        batch = posterior_samples[i:min(i+batch_size, n_samples)]
        acf_theta = batch[:, :2]
        marginal_theta = jax.vmap(convert_3_to_4_param_nig)(batch[:, 2:])
        
        key, *keys = jax.random.split(key, len(batch) + 1)
        
        sims, _ = jax.jit(jax.vmap(slice_sample_sup_ig_nig_trawl, in_axes=(None, None, 0, 0, 0)), 
                         static_argnums=(0, 1))(observed_trawl.shape[-1], 1.0, acf_theta, marginal_theta, jnp.array(keys))
        
        emp_acf = jnp.array([compute_empirical_acf(t, adjusted=True, nlags=nlags) for t in sims])
        theo_acf = jax.vmap(lambda p: corr_sup_ig_envelope(jnp.arange(nlags+1), p))(acf_theta)
        all_diff.append(emp_acf - theo_acf)
    
    diff = jnp.concatenate(all_diff, axis=0)
    return jnp.mean(diff, axis=0), jnp.percentile(diff, 2.5, axis=0), jnp.percentile(diff, 97.5, axis=0)
    
    
if __name__ == '__main__':
    
    observed_trawl = jnp.ones(1250)
    MAP = jnp.array([15., 16., 0.5, 1.25, 3])
    key = jax.random.PRNGKey(3241)
    num_replicated_for_bias_estimation = 1000
    nlags = 40    
    mean_bias, lower_ci, upper_ci = estimate_bias_from_MAP(MAP, observed_trawl, num_replicated_for_bias_estimation, nlags, key)
    print(f"Mean bias shape: {mean_bias.shape}")
    print(f"Mean bias at lag 1: {mean_bias[1]:.4f}")
    print(f"95% CI at lag 1: [{lower_ci[1]:.4f}, {upper_ci[1]:.4f}]")
    
    mean_bias, lower_ci, upper_ci = estimate_bias_from_MAP(MAP, observed_trawl, num_replicated_for_bias_estimation, nlags, key)
    a,b,c = estimate_bias_from_posterior_samples_batched(jnp.tile(MAP[jnp.newaxis,:], (1000,1)), observed_trawl, nlags, key, batch_size=100)
    print(a,b,c)
