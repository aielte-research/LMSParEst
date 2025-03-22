import numpy as np
from scipy.linalg import toeplitz, cholesky, solve_toeplitz
from scipy.optimize import fminbound

def autocorr_vector(H, N):
    """
    Computes the autocorrelation function for fGn at lags 0,...,N-1 using
    the formula: ρ(k) = 0.5 * [ (k+1)^(2H) - 2*k^(2H) + |k-1|^(2H) ].
    """
    lags = np.arange(N)
    r = 0.5 * ((lags + 1)**(2 * H) - 2 * (lags)**(2 * H) + np.abs(lags - 1)**(2 * H))
    return r

def log_posterior_toeplitz(H, x):
    """
    Computes the (unnormalized) log-posterior for a candidate H given a centered time series x,
    using a Toeplitz-based approach.
    """
    N = len(x)
    # Compute the autocorrelation vector and construct the Toeplitz matrix R implicitly.
    r = autocorr_vector(H, N)
    # Use Cholesky factorization to obtain the log-determinant.
    try:
        L = cholesky(toeplitz(r), lower=True)
    except np.linalg.LinAlgError:
        return -np.inf
    logdet = 2 * np.sum(np.log(np.diag(L)))
    
    # Define vector of ones.
    e = np.ones(N)
    # Solve R y = e and R z = x using the Toeplitz solver.
    y = solve_toeplitz((r, r), e)
    z = solve_toeplitz((r, r), x)
    
    B = e @ y
    A = e @ z
    # Safeguard: if the arguments for the logarithms are not positive, return -∞.
    if A - B**2 <= 0 or B <= 0:
        return -np.inf
    log_post = 0.5 * logdet - ((N - 1) / 2) * np.log(A - B**2) + ((N - 2) / 2) * np.log(B)
    return log_post

def bayesian_hk_estimate_toeplitz(x, n_samples=500):
    """
    Estimates the Hurst exponent of an fGn time series x using the Bayesian HK method with a Toeplitz-based solver.
    Uses fminbound to find the maximum (via minimizing the negative log-posterior) so that an appropriate constant M is determined for accept–reject sampling.
    Returns the accepted posterior samples and the median as the point estimate.
    """
    # Center the time series.
    x = x - np.mean(x)
    N = len(x)
    
    # Define an objective function for fminbound (minimizing the negative log-posterior).
    def neg_log_post(H):
        return -log_posterior_toeplitz(H, x)
    
    # Use fminbound to find the H that minimizes the negative log-posterior.
    H_opt = fminbound(neg_log_post, 0, 1)
    # Determine M as the maximum unnormalized posterior value.
    M = np.exp(log_posterior_toeplitz(H_opt, x))
    
    accepted = []
    n_iter = 0
    # Accept–reject sampling with candidates drawn uniformly over (0,1).
    while len(accepted) < n_samples:
        H_cand = np.random.uniform(0, 1)
        log_post_cand = log_posterior_toeplitz(H_cand, x)
        f_cand = np.exp(log_post_cand) if log_post_cand > -np.inf else 0.0
        u = np.random.uniform(0, M)
        if u <= f_cand:
            accepted.append(H_cand)
        n_iter += 1
        if n_iter > 1_000_000:
            print("Warning: Too many iterations; sampler may be inefficient.")
            break
    accepted = np.array(accepted)
    median_est = np.median(accepted)
    return median_est
    
# Example usage:
if __name__ == '__main__':
    from whittlehurst import fbm

    # Original Hurst value to test with
    Hs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for H in Hs:

        # Generate an fBm realization
        seq = fbm(H=H, n=3200)

        # Calculate the increments, because the estimator works with the fGn spectrum
        seq = np.diff(seq)

        # Estimate the Hurst exponent
        H_est = bayesian_hk_estimate_toeplitz(seq)

        print(H, H_est)