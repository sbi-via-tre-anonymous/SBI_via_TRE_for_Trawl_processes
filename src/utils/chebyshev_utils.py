import jax.numpy as jnp
from jax.numpy.fft import fft
from jax.numpy.fft import ifft
import jax
import numpy as np
from functools import partial


def interpolation_points_domain(N, a, b):
    """
    Generate N Chebyshev points directly in the domain [a,b], boundaries included.

    Parameters:
    -----------
    N : int
        Number of points
    a, b : float
        Domain boundaries

    Returns:
    --------
    jnp.ndarray : Array of N Chebyshev points in [a,b]
    """
    if N == 1:
        return jnp.array([(a + b) / 2.0])

    # Generate standard Chebyshev points in [-1,1]
    standard_points = jnp.cos(jnp.arange(N) * jnp.pi / (N - 1))

    # Map to domain [a,b]
    return 0.5 * (a + b) + 0.5 * (b - a) * standard_points


def sample_function_domain(f, N, a, b):
    """
    Sample a function directly on N Chebyshev points in domain [a,b].

    Parameters:
    -----------
    f : function
        The function to sample
    N : int
        Number of points
    a, b : float
        Domain boundaries

    Returns:
    --------
    jnp.ndarray : Function values at Chebyshev points in [a,b]
    """
    x = interpolation_points_domain(N, a, b)
    return f(x)


def even_data(data):
    """
    Construct Extended Data Vector (equivalent to creating an
    even extension of the original function)
    Return: array of length 2(N-1)
    For instance, [0,1,2,3,4] --> [0,1,2,3,4,3,2,1]
    """
    return jnp.concatenate([data, data[-2:0:-1]])


def dct_via_fft(data: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a DCT-like transform using FFT, JAX version.
    """
    N = data.shape[0] // 2
    fftdata = fft(data, axis=0)[:N + 1] / N
    fftdata = fftdata.at[0].set(fftdata[0] / 2.)
    fftdata = fftdata.at[-1].set(fftdata[-1] / 2.)

    result = jnp.real(fftdata) if jnp.isrealobj(data) else fftdata
    return result


@jax.jit
def polyfit_domain(sampled, a, b):
    """
    Compute Chebyshev coefficients for values located on Chebyshev points in [a,b].

    Parameters:
    -----------
    sampled : jnp.ndarray
        Function values at Chebyshev points in [a,b]
    a, b : float
        Domain boundaries

    Returns:
    --------
    jnp.ndarray : Chebyshev coefficients
    """
    asampled = jnp.asarray(sampled)
    if len(asampled) == 1:
        return asampled

    evened = even_data(asampled)
    coeffs = dct_via_fft(evened)

    # Note: coefficients are already in the correct form for the specified domain
    return coeffs


vec_polyfit_domain = jax.jit(jax.vmap(polyfit_domain, in_axes=(0, None, None)))


def chebval_ab_for_one_x(x, coeff, a, b):
    """
    Evaluate a Chebyshev series at one point x in domain [a,b] using coefficients.
    Works directly with x in domain [a,b] without requiring a separate mapping step.

    Parameters:
    -----------
    x : float
        Point in domain [a,b] at which to evaluate
    coeff : jnp.ndarray
        Chebyshev coefficients
    a, b : float
        Domain boundaries

    Returns:
    --------
    float : The value of the Chebyshev polynomial at point x
    """
    if len(coeff) == 1:
        return coeff[0]

    # Map x from [a,b] to [-1,1] within the algorithm
    z = (2*x - (a+b))/(b-a)

    b0 = jnp.zeros_like(z)
    b1 = jnp.zeros_like(z)

    def body_fun(carry, cj):
        b0, b1 = carry
        new_b0 = cj + 2 * z * b0 - b1
        new_b1 = b0
        return (new_b0, new_b1), None

    (b0, b1), _ = jax.lax.scan(
        body_fun,
        (b0, b1),
        jnp.flip(coeff[1:]),
        reverse=False
    )

    return coeff[0] + z * b0 - b1


# Vectorized version that works directly with domain [a,b]
chebval_ab_jax = jax.jit(
    jax.vmap(chebval_ab_for_one_x, in_axes=(0, None, None, None)))

vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes = jax.jit(
    jax.vmap(chebval_ab_for_one_x, in_axes=(0, 0, None, None)))


def chebint_ab(coeff, a, b):
    """
    Integrate a Chebyshev series defined on domain [a,b] once.

    Parameters:
    -----------
    c : jnp.ndarray
        Chebyshev coefficients
    a, b : float
        Domain boundaries

    Returns:
    --------
    jnp.ndarray : Chebyshev coefficients of the integral
    """
    coeff = jnp.atleast_1d(coeff)
    n = coeff.shape[0]

    # Initialize with zeros
    out = jnp.zeros(n + 1, dtype=coeff.dtype)

    # Scale factor for the domain [a,b]
    scale = (b - a) / 2.0

    # First coefficient term T_0 -> T_1
    out = out.at[1].set(coeff[0] * scale)

    if n > 1:
        out = out.at[2].set(coeff[1] * scale / 4)

    def body(j, out):
        coeffj = coeff[j] * scale
        out = out.at[j + 1].set(coeffj / (2 * (j + 1)))
        out = out.at[j - 1].add(-coeffj / (2 * (j - 1)))
        return out

    out = jax.lax.fori_loop(2, n, body, out)

    return out


def get_coeffs(f, a, b, N):

    # Sample function directly on domain [a,b]
    sampled = sample_function_domain(f, N+1, a, b)

    # Compute coefficients
    coeffs = polyfit_domain(sampled, a, b)

    return coeffs


@jax.jit
def integrate_from_sampled(sampled, a, b):

    coeffs = polyfit_domain(sampled, a, b)
    coeffs_int = chebint_ab(coeffs, a, b)

    # Evaluate antiderivative at domain endpoints
    endpoints = jnp.array([a, b])
    results = chebval_ab_jax(endpoints, coeffs_int, a, b)

    # Return the difference
    return results[1] - results[0]


vec_integrate_from_samples = jax.jit(
    jax.vmap(integrate_from_sampled, in_axes=(0, None, None)))


def integrate_ab(f, a, b, N):
    """
    Compute the integral of function f over the entire domain [a,b].
    Uses direct [a,b] domain functions throughout.

    Parameters:
    -----------
    f : function
        The function to integrate
    a, b : float
        Domain boundaries
    N : int
        Number of Chebyshev points to use

    Returns:
    --------
    float : The approximate value of the integral
    """
    # Sample function directly on domain [a,b]
    sampled = sample_function_domain(f, N+1, a, b)

    # Compute coefficients
    coeffs = polyfit_domain(sampled, a, b)

    # Compute indefinite integral coefficients
    coeffs_int = chebint_ab(coeffs, a, b)

    # Evaluate antiderivative at domain endpoints
    endpoints = jnp.array([a, b])
    results = chebval_ab_jax(endpoints, coeffs_int, a, b)

    # Return the difference
    return results[1] - results[0]


def integrate_subinterval_ab(f, a, b, c, d, N):
    """
    Compute the integral of function f over subinterval [c,d] ⊆ [a,b].
    Uses direct [a,b] domain functions throughout.

    Parameters:
    -----------
    f : function
        The function to integrate
    a, b : float
        Original domain boundaries
    c, d : float
        Subinterval boundaries where c >= a and d <= b
    N : int
        Number of Chebyshev points to use

    Returns:
    --------
    float : The approximate value of the integral of f over [c,d]
    """
    # Verify that the subinterval is contained in the original domain
    if c < a or d > b:
        raise ValueError(
            f"Subinterval [{c},{d}] must be contained in original domain [{a},{b}]")
        return jnp.nan

    # Sample function directly on domain [a,b]
    sampled = sample_function_domain(f, N+1, a, b)

    # Compute coefficients
    coeffs = polyfit_domain(sampled, a, b)

    # Compute indefinite integral coefficients
    coeffs_int = chebint_ab(coeffs, a, b)

    # Evaluate antiderivative at subinterval endpoints
    endpoints = jnp.array([c, d])
    results = chebval_ab_jax(endpoints, coeffs_int, a, b)

    # Return the difference
    return results[1] - results[0]

###### SAMPLING ######


@partial(jax.jit, static_argnames=('nr_samples',))
def sample_from_coeff(coeff, key, a, b,  nr_samples):

    # np.random.uniform(0, 1, nr_samples)
    unif_samples = jax.random.uniform(key, shape=nr_samples)
    cdf = chebcdf(coeff, a, b)
    def shifted(x): return cdf(x) - unif_samples

    # if num_samples == 1:
    #    samples = scipy.optimize.bisect(shifted, lower_bd, upper_bd)
#
    # elif num_samples > 1:
    samples = vectorized_bisection(shifted, a, b)
    return samples


vec_sample_from_coeff = jax.jit(jax.vmap(
                                sample_from_coeff, in_axes=(0, 0, None, None, None)),
                                static_argnames=('nr_samples',))


def chebcdf(coeff, a, b):

    coeffs_int = chebint_ab(coeff, a, b)
    endpoints = jnp.array([a, b])
    results = chebval_ab_jax(endpoints, coeffs_int, a, b)
    offset = results[0]
    scale = results[1] - results[0]

    return lambda x: (chebval_ab_for_one_x(x, coeffs_int, a, b) - offset) / scale


def vectorized_bisection(func, lower, upper, max_iter=52):

    # f = vectorized_grad_log_density_dictionary[distr_name]

    for _iter in range(max_iter):

        mid = (lower + upper) / 2

        mid_sgn = jnp.sign(func(mid))
        ub_sgn = jnp.sign(func(upper))
        lb_sgn = jnp.sign(func(lower))

        lower = jnp.where(lb_sgn == mid_sgn, mid, lower)
        upper = jnp.where(ub_sgn == mid_sgn, mid, upper)

    mid = (lower + upper) / 2
    return mid


# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy
    # Test function

    def test_function(x):
        return - 15 * jnp.sin(x) + x**2 + jnp.exp(x/4)

    def test_function2(x):
        return 2 * jnp.sin(x) + x**2 + jnp.exp(x/4)

    # Original domain
    a, b = 0.0, 9.0

    # Subinterval
    c, d = 4.0, 7.0

    # Number of Chebyshev points
    N = 6

    # Compute integrals
    full_integral = integrate_ab(test_function, a, b, N)
    subinterval_integral = integrate_subinterval_ab(
        test_function, a, b, c, d, N)

    print(f"Integral over full domain [{a},{b}] ≈ {full_integral}")
    print(f"Integral over subinterval [{c},{d}] ≈ {subinterval_integral}")

    scipy.integrate.quad(test_function, a, b)
    scipy.integrate.quad(test_function, c, d)

    test_values = jnp.linspace(a, b, 100)
    coeff = get_coeffs(test_function, a, b, N)
    approx_poly = chebval_ab_jax(test_values, coeff, a, b)

    plt.plot(test_values, test_function(test_values),
             label='test_function', alpha=0.5, marker='x')
    plt.plot(test_values, approx_poly, label='approx', alpha=0.5, marker='o')
    plt.show()
    plt.legend()

    ###### SAMPLING ######
