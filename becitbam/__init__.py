import numpy as np
import scipy as sp
import scipy.optimize

def _check_arguments(s,a,mu=None,alpha=None):
    A=np.sum(a)
    assert (a>=0).all()
    assert 0 <= s <=A
    if mu is not None:
        assert 0 < mu < A
    if alpha is not None:
        assert 0 < alpha < 1

r'''

      _               _                            _ _
  ___| | __ _ ___ ___(_) ___   _ __ ___  ___ _   _| | |_ ___
 / __| |/ _` / __/ __| |/ __| | '__/ _ \/ __| | | | | __/ __|
| (__| | (_| \__ \__ \ | (__  | | |  __/\__ \ |_| | | |_\__ \
 \___|_|\__,_|___/___/_|\___| |_|  \___||___/\__,_|_|\__|___/

'''

def hoeffding_thm1(s,n,mu):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,1]
        sum E[X_i] = mu

    Returns upper bound on log P(S>=s).
    '''

    if s<=mu:
        return 0.0
    elif s>n:
        return -np.inf

    hoeffding_mu = mu/n
    hoeffding_t = s/n - mu/n

    assert hoeffding_t>=0
    assert s<=n
    assert mu>=0
    assert mu<=n

    T1 = (hoeffding_mu/(hoeffding_mu+hoeffding_t))**(hoeffding_mu+hoeffding_t)

    if 1-hoeffding_mu-hoeffding_t!=0:
        T2 = ((1-hoeffding_mu)/(1-hoeffding_mu-hoeffding_t))**(1-hoeffding_mu-hoeffding_t)
    else:
        T2 = 1.0

    return n*np.log(T1*T2)

def hoeffding_thm1_rescaled(s,n,mu,a):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a]
        sum E[X_i] = mu

    Returns upper bound on log P(S>=s)
    '''

    '''
        P(S > s) = P(sum_i X_i > s)  subject to sum E[X_i] = mu
                 = P(sum_i X_i/a > s/a) subject to sum E[X_i/a] = mu/a
    '''

    a=float(a)

    if a<0:
        raise ValueError("a should be positive")

    return hoeffding_thm1(s/a,n,mu/a)


def hoeffding_thm2(s,mu,a):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]
        sum E[X_i] = mu

    Returns upper bound on log P(S>=s)
    '''

    _check_arguments(s,a,mu=mu)

    hoeffding_t = s - mu

    if s<=mu:
        return 0.0
    else:
        hoeffding_t = s-mu
        return -2*hoeffding_t*hoeffding_t / np.sum(a**2)


def bentkus(s, mu, a):
    '''
    Bentkus bound for P(S >= s) by rescaling to maximum interval width = 1.

    Based on Bentkus (2004) Theorem 1.2: For independent random variables X_i in [0, 1]
    with total mean mu, we have P(S >= s) <= e * P°(Binomial(n, p) >= s) where p = mu/n.

    This function rescales the problem so that max(a_i) = 1, then applies the bound.
    For non-integer s (after rescaling), log-linear interpolation is used between
    adjacent integer points as specified in the Bentkus paper (equation 1.8):
        B°(z) = B^(1-λ)(x) * B^λ(y)  where z = (1-λ)x + λy

    Let

        S = sum_i^n X_i
        X_i in [0, a_i]
        sum E[X_i] = mu

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
    '''
    from scipy.stats import binom

    a = np.asarray(a)
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

    # Handle edge cases
    if s_rescaled > n:
        return 0.0
    if s_rescaled <= 0:
        return 1.0

    # Get the floor and ceiling integer points for interpolation
    x = int(np.floor(s_rescaled))  # lower integer point
    y = int(np.ceil(s_rescaled))   # upper integer point

    # If s_rescaled is exactly an integer, no interpolation needed
    if x == y or s_rescaled == x:
        if x <= 0:
            binomial_tail = 1.0
        elif x > n:
            binomial_tail = 0.0
        else:
            binomial_tail = binom.sf(x - 1, n, p)
    else:
        # Log-linear interpolation: B°(z) = B^(1-λ)(x) * B^λ(y)
        # where z = (1-λ)x + λy, so λ = (z - x) / (y - x) = z - x (since y - x = 1)
        lam = s_rescaled - x

        # Get binomial survival function at integer points x and y
        if x <= 0:
            B_x = 1.0
        else:
            B_x = binom.sf(x - 1, n, p)

        if y > n:
            B_y = 0.0
        else:
            B_y = binom.sf(y - 1, n, p)

        # Log-linear interpolation: B°(z) = B_x^(1-λ) * B_y^λ
        # Handle edge cases where B_x or B_y is 0
        if B_y <= 0:
            # If B_y is 0, then B°(z) should be 0 for any λ > 0
            binomial_tail = 0.0
        elif B_x <= 0:
            # This shouldn't happen if x > 0, but handle it
            binomial_tail = 0.0
        else:
            # B°(z) = B_x^(1-λ) * B_y^λ = exp((1-λ)*log(B_x) + λ*log(B_y))
            binomial_tail = np.exp((1 - lam) * np.log(B_x) + lam * np.log(B_y))

    # Bentkus constant is e ≈ 2.72
    bentkus_prob = np.e * binomial_tail

    # Probability cannot exceed 1
    return min(bentkus_prob, 1.0)


def bentkus_binomial(s, mu, a):
    '''
    Bentkus bound for P(S >= s) using the symmetric Bernoulli construction
    with RMS scale (Corollary 1.4 / Theorem 1.3).

    This function uses the root-mean-square scale a = sqrt(sum(a_i^2)/n) 
    instead of the maximum scale. The bound compares the centered variable 
    M = S - mu to a sum of symmetric Bernoulli random variables.

    Let
        S = sum_i^n X_i
        X_i in [0, a_i]
        sum E[X_i] = mu

    We bound P(S >= s) by analyzing the centered variable M = S - mu.
    With x = s - mu, Corollary 1.4 gives:
        P(M >= x) <= (2e^3/9) * P°(S_n >= x)

    where S_n = sum of n i.i.d. symmetric Bernoulli variables epsilon_i in {-a, +a}
    with a = sqrt(sum(a_i^2)/n) being the RMS scale.

    Since S_n = a(2K - n) where K ~ Binomial(n, 1/2), the lattice points are
    at y = a(2k - n) for k = 0, 1, ..., n. For non-lattice points, log-linear
    interpolation is used.

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
        Bentkus binomial bound on P(S >= s) (probability, not log probability)
    '''
    from scipy.stats import binom

    a = np.asarray(a)
    n = len(a)
    
    # Handle edge case of no variables
    if n == 0:
        return 1.0 if s <= 0 else 0.0
    
    # RMS scale: a_rms = sqrt(sum(a_i^2) / n)
    a_rms = np.sqrt(np.sum(a ** 2) / n)
    
    # Handle edge case where all a_i = 0
    if a_rms <= 0:
        return 1.0 if s <= 0 else 0.0
    
    # Centered threshold: x = s - mu
    x = s - mu
    
    # If x <= -n*a_rms (i.e., s <= mu - n*a_rms), probability is 1
    # If x > n*a_rms (i.e., s > mu + n*a_rms), probability is 0
    if x <= -n * a_rms:
        return 1.0
    if x > n * a_rms:
        return 0.0
    
    # The symmetric Bernoulli sum S_n takes values a(2k-n) for k = 0, 1, ..., n
    # Lattice spacing is 2*a_rms
    # S_n = a_rms * (2K - n) where K ~ Binomial(n, 1/2)
    # P(S_n >= x) = P(K >= (x/a_rms + n)/2)
    
    # Find k such that a_rms*(2k - n) is the lattice point at or just below x
    # Solve: a_rms*(2k - n) <= x => k <= (x/a_rms + n) / 2
    k_continuous = (x / a_rms + n) / 2.0
    
    # k must be in [0, n], and we need k such that a_rms*(2k-n) <= x
    k = int(np.floor(k_continuous))
    k = max(0, min(n, k))
    
    # Lattice point y = a_rms*(2k - n)
    y = a_rms * (2 * k - n)
    
    # Next lattice point y + 2*a_rms = a_rms*(2(k+1) - n)
    y_next = a_rms * (2 * (k + 1) - n)
    
    # Check if x is at a lattice point (within numerical tolerance)
    if np.abs(x - y) < 1e-12 * max(1.0, abs(x), abs(y)):
        # x is exactly at lattice point y = a_rms*(2k - n)
        # P(S_n >= y) = P(K >= k)
        if k <= 0:
            tail_prob = 1.0
        elif k > n:
            tail_prob = 0.0
        else:
            # P(K >= k) = 1 - P(K <= k-1) = binom.sf(k-1, n, 0.5)
            tail_prob = binom.sf(k - 1, n, 0.5)
    else:
        # x is between lattice points y and y_next
        # Log-linear interpolation:
        # lambda = (x - y) / (2*a_rms) in (0, 1)
        # P°(S_n >= x) = P(K >= k)^(1-lambda) * P(K >= k+1)^lambda
        
        lam = (x - y) / (2 * a_rms)
        lam = np.clip(lam, 0.0, 1.0)  # numerical safety
        
        # P(K >= k) 
        if k <= 0:
            P_k = 1.0
        elif k > n:
            P_k = 0.0
        else:
            P_k = binom.sf(k - 1, n, 0.5)
        
        # P(K >= k+1)
        if k + 1 <= 0:
            P_k1 = 1.0
        elif k + 1 > n:
            P_k1 = 0.0
        else:
            P_k1 = binom.sf(k, n, 0.5)
        
        # Log-linear interpolation
        if P_k1 <= 0:
            # If P(K >= k+1) = 0, result is 0 for any lambda > 0
            tail_prob = 0.0
        elif P_k <= 0:
            # This shouldn't happen for valid k, but handle it
            tail_prob = 0.0
        else:
            # P°(S_n >= x) = P_k^(1-lambda) * P_k1^lambda
            tail_prob = np.exp((1 - lam) * np.log(P_k) + lam * np.log(P_k1))
    
    # Apply the prefactor 2*e^3/9
    prefactor = 2 * np.e ** 3 / 9
    bentkus_prob = prefactor * tail_prob
    
    # Probability cannot exceed 1
    return min(bentkus_prob, 1.0)


r'''
 _   _       _     _          _                            __  __
| |_(_) __ _| |__ | |_    ___| |__   ___ _ __ _ __   ___  / _|/ _|
| __| |/ _` | '_ \| __|  / __| '_ \ / _ \ '__| '_ \ / _ \| |_| |_
| |_| | (_| | | | | |_  | (__| | | |  __/ |  | | | | (_) |  _|  _|
 \__|_|\__, |_| |_|\__|  \___|_| |_|\___|_|  |_| |_|\___/|_| |_|
       |___/
'''

def uniqify(a):
    a=a[a!=0]
    return np.unique(a,return_counts=True)

def _lamstar(mu,asrt,w,t):
    b=(np.exp(asrt*t) -1) / asrt

    def meandiff(lam):
        return np.sum(_taustar(asrt,b,lam)*w) - mu

    LB=np.min((np.exp(asrt*t)-1)/(np.exp(asrt*t)-1+asrt))
    UB=np.max(b)

    return sp.optimize.bisect(meandiff,LB,UB)

def _taustar(asrt,b,lam):
    tau = (b - lam)/(b*lam)
    tau = np.clip(tau,0,asrt)
    return tau

def _g_weighted(s,mu,asrt,w,t,lam):
    assert t>=0 and lam>=0

    if t==0:
        return lam*mu - s*t
    if lam==0:
        return (np.sum(asrt*w)-s)*t

    b=(np.exp(asrt*t) -1) / asrt
    tau = (b - lam)/(b*lam)
    tau = np.clip(tau,0,asrt)

    rez=np.sum(np.log(1+b*tau)*w) + lam*(mu - np.sum(tau*w)) - s*t

    return rez

def sharp_chernoff(s,mu,a):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]
        sum E[X_i] = mu

    Returns upper bound on log P(S>=s)
    '''
    _check_arguments(s,a,mu=mu)
    asrt,w=uniqify(a)

     # keep a in check to maintain numerical stability
    mx=asrt.max()
    s=s/mx
    mu=mu/mx
    asrt=asrt/mx

    return _sharp_chernoff_weighted(s,mu,asrt,w).fun

def _sharp_chernoff_weighted(s,mu,asrt,w):
    # find reasonable initialization
    t_init=.5
    lam_init=_lamstar(mu,asrt,w,t_init)

    rez=sp.optimize.minimize(
        lambda tlam: _g_weighted(s,mu,asrt,w,tlam[0],tlam[1]),
        (t_init,lam_init),
        bounds=[(0,np.inf),(0,np.inf)],
        method='nelder-mead'
    )
    if not rez.success:
        raise Exception("Minimization failed!")

    return rez

def calculate_sharp_chernoff_parameters(s, mu, a):
    '''
    Compute the optimal Chernoff parameters for the sharp bound.

    Let

        S = sum_i^n X_i
        X_i in [0, a_i]
        sum E[X_i] = mu

    Returns the optimal tstar >= 0 and parameters tau such that if

        X_i ~ Bernoulli(tau_i / a_i) * a_i

    then E[exp(tstar * sum X_i)] * exp(-tstar * s) achieves the tight
    Chernoff bound. Here tau_i = E[X_i] is the mean of X_i.

    Parameters
    ----------
    s : float
        The threshold value
    mu : float
        The mean of the sum
    a : numpy.ndarray
        Upper bounds for each variable (X_i in [0, a_i])

    Returns
    -------
    tstar : float
        The optimal tilting parameter (>= 0)
    tau : numpy.ndarray
        Mean parameters for each element in a (same shape as a).
        tau_i = E[X_i], so tau_i is in [0, a_i].
        For elements where a_i = 0, tau_i = 0.
    '''
    _check_arguments(s, a, mu=mu)

    a_original = np.asarray(a)

    # Identify zero and non-zero indices
    zero_mask = (a_original == 0)
    nonzero_indices = np.where(~zero_mask)[0]
    a_nonzero = a_original[~zero_mask]

    # Get unique values and inverse mapping for non-zero elements
    asrt, inverse_indices, w = np.unique(a_nonzero, return_inverse=True, return_counts=True)

    # Normalize for numerical stability (same as in sharp_chernoff)
    mx = asrt.max()
    s_norm = s / mx
    mu_norm = mu / mx
    asrt_norm = asrt / mx

    # Get optimal t and lambda (on normalized scale)
    rez = _sharp_chernoff_weighted(s_norm, mu_norm, asrt_norm, w)
    tstar_norm = rez.x[0]
    lamstar = rez.x[1]

    # Convert tstar back to original scale
    # Since s_norm = s/mx and a_norm = a/mx, the optimal t for normalized
    # problem relates to the original by: t_orig = t_norm / mx
    tstar = tstar_norm / mx

    # Compute tau for unique values
    # _taustar returns E[X_i] (mean), clipped to [0, a_i]
    # tau_i = E[X_i] is what we want to return
    if tstar_norm == 0:
        # When t=0, the Chernoff bound is 1 regardless of the distribution,
        # so any tau values satisfying the mean constraint are valid.
        # We return zeros as a convention (the bound is achieved for any distribution).
        tau_unique = np.zeros_like(asrt_norm)
    else:
        b = (np.exp(asrt_norm * tstar_norm) - 1) / asrt_norm
        tau_unique = _taustar(asrt_norm, b, lamstar)

    # Scale tau back to original scale (tau was computed on normalized a)
    tau_unique_scaled = tau_unique * mx

    # Un-uniqify: map tau values back to original array shape
    tau = np.zeros_like(a_original, dtype=float)
    # Use inverse_indices to map unique tau values back to non-zero positions
    tau[~zero_mask] = tau_unique_scaled[inverse_indices]
    # Zero positions already have tau=0 from initialization

    return tstar, tau

def wpb_chernoff_tails(s, tau, a, t):
    '''
    Compute the log Chernoff bound for a weighted Poisson binomial distribution.

    Let X_i ~ Bernoulli(tau_i / a_i) * a_i where tau_i = E[X_i]. Returns:

        log(E[exp(t * sum X_i)] * exp(-t * s))

    Parameters
    ----------
    s : float
        The threshold value
    tau : numpy.ndarray
        Mean parameters for each variable (tau_i = E[X_i], in [0, a_i])
    a : numpy.ndarray
        Weights/upper bounds for each variable (X_i in {0, a_i})
    t : float
        The tilting parameter

    Returns
    -------
    float
        The log Chernoff bound value
    '''
    tau = np.asarray(tau)
    a = np.asarray(a)

    assert len(tau) == len(a)
    assert (tau >= 0).all()
    assert (a >= 0).all()
    assert t >= 0
    # tau_i should be <= a_i (mean can't exceed upper bound)
    assert (tau <= a + 1e-10).all()  # small tolerance for numerical precision

    # q_i = tau_i / a_i is the Bernoulli probability
    # E[exp(t * X_i)] = (1 - q_i) + q_i * exp(t * a_i)
    # Use log-sum-exp trick for numerical stability when t*a is large
    log_mgf_terms = []
    for i in range(len(tau)):
        if a[i] == 0:
            log_mgf_terms.append(0.0)  # X_i = 0 always, so E[exp(t*X_i)] = 1
        elif tau[i] == 0:
            log_mgf_terms.append(0.0)  # q_i = 0, so E[exp(t*X_i)] = 1
        elif tau[i] >= a[i] - 1e-10:  # q_i ≈ 1
            log_mgf_terms.append(t * a[i])  # E[exp(t*X_i)] = exp(t*a_i)
        else:
            q_i = tau[i] / a[i]
            ta = t * a[i]
            if ta > 100:  # exp(ta) would overflow, use asymptotic
                log_mgf_terms.append(ta + np.log(q_i))
            else:
                log_mgf_terms.append(np.log((1 - q_i) + q_i * np.exp(ta)))

    log_mgf_sum = np.sum(log_mgf_terms)
    return log_mgf_sum - t * s

def wpb_exact_tails(s, tau, a):
    '''
    Compute the log of the exact tail probability for a weighted Poisson binomial distribution.

    Let X_i ~ Bernoulli(tau_i / a_i) * a_i where tau_i = E[X_i]. Returns:

        log(P(sum X_i >= s))

    This is computed using dynamic programming (convolution of the distributions).

    Parameters
    ----------
    s : float
        The threshold value
    tau : numpy.ndarray
        Mean parameters for each variable (tau_i = E[X_i], in [0, a_i])
    a : numpy.ndarray
        Weights/upper bounds for each variable (X_i in {0, a_i})

    Returns
    -------
    float
        The log of the exact tail probability log(P(sum X_i >= s))
    '''
    tau = np.asarray(tau)
    a = np.asarray(a)

    assert len(tau) == len(a)
    assert (tau >= 0).all()
    assert (a >= 0).all()
    # tau_i should be <= a_i (mean can't exceed upper bound)
    assert (tau <= a + 1e-10).all()  # small tolerance for numerical precision

    n = len(tau)
    if n == 0:
        return 0.0 if s <= 0 else -np.inf

    # Use dynamic programming with a dictionary to track (value, probability) pairs
    # Start with the distribution of 0 (probability 1)
    dist = {0.0: 1.0}

    for i in range(n):
        new_dist = {}
        a_i = a[i]

        # q_i = tau_i / a_i is the Bernoulli probability
        if a_i == 0:
            q_i = 0.0  # X_i = 0 always
        else:
            q_i = tau[i] / a_i

        for val, prob in dist.items():
            # X_i = 0 with probability (1 - q_i)
            v0 = val
            if v0 in new_dist:
                new_dist[v0] += prob * (1 - q_i)
            else:
                new_dist[v0] = prob * (1 - q_i)

            # X_i = a_i with probability q_i
            v1 = val + a_i
            if v1 in new_dist:
                new_dist[v1] += prob * q_i
            else:
                new_dist[v1] = prob * q_i

        dist = new_dist

    # Sum probabilities for all values >= s
    tail_prob = sum(prob for val, prob in dist.items() if val >= s)
    if tail_prob <= 0:
        return -np.inf
    return np.log(tail_prob)

r'''

     _           _
 ___| |__   __ _| |_ ___
/ __| '_ \ / _` | __/ __|
\__ \ | | | (_| | |_\__ \
|___/_| |_|\__,_|\__|___/

'''

def shat_from_sharp_chernoff(mu,a,alpha):
    '''
    Returns biggest s so that inf_t E_mu[exp(St-st)]<alpha
    '''

    asrt,w=uniqify(a)

     # keep a in check to maintain numerical stability
    mx=asrt.max()
    mu=mu/mx
    asrt=asrt/mx
    A=np.sum(asrt*w)

    def f(s):
        return _sharp_chernoff_weighted(s,mu,asrt,w).fun-np.log(alpha)

    if f(A)>0.0:
        return A

    UB=A-(A-mu)*.5
    while f(UB)>0:
        UB=A-(A-UB)*.5

    return sp.optimize.bisect(f,mu,UB)*mx

def shat_from_hoeffding_thm1(n,mu,alim,alpha):
    '''
    Returns biggest s so that inf_t E_mu[exp(St-st)]<alpha
    '''

    def f(s):
        return hoeffding_thm1_rescaled(s,n,mu,alim)-np.log(alpha)

    A=alim*n

    if f(A)>0.0:
        return A

    UB=A-(A-mu)*.5
    while f(UB)>0:
        UB=A-(A-UB)*.5

    return sp.optimize.bisect(f,mu,UB)


r'''
                  __ _     _
  ___ ___  _ __  / _(_) __| | ___ _ __   ___ ___
 / __/ _ \| '_ \| |_| |/ _` |/ _ \ '_ \ / __/ _ \
| (_| (_) | | | |  _| | (_| |  __/ | | | (_|  __/
 \___\___/|_| |_|_| |_|\__,_|\___|_| |_|\___\___|

 _       _                       _
(_)_ __ | |_ ___ _ ____   ____ _| |___
| | '_ \| __/ _ \ '__\ \ / / _` | / __|
| | | | | ||  __/ |   \ V / (_| | \__ \
|_|_| |_|\__\___|_|    \_/ \__,_|_|___/

'''


def confidence_from_hoeffding_thm2(s,a,alpha):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]

    Returns mu so that P(S>=s)<=alpha as long as E[S]=mu.
    '''

    _check_arguments(s,a,alpha=alpha)
    return s- np.sqrt(-.5*np.log(alpha)*np.sum(a**2))

def confidence_from_hoeffding_thm1(s,n,alim,alpha):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,alim]

    Returns mu so that P(S>=s)<=alpha as long as E[S]=mu.
    '''

    def f(mu):
        return hoeffding_thm1_rescaled(s,n,mu,alim)-np.log(alpha)

    LB=s*.5
    while f(LB)>0:
        LB=LB*.5

    return sp.optimize.bisect(f,LB,s)


def confidence_from_sharp_chernoff(s,a,alpha):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]

    Returns mu so that P(S>=s)<=alpha as long as E[S]=mu.

    Calculated by directly inverting sharp_chernoff.
    '''

    _check_arguments(s,a,alpha=alpha)

    asrt,w=uniqify(a)

     # keep a in check to maintain numerical stability
    mx=asrt.max()
    s=s/mx
    asrt=asrt/mx

    def f(mu):
        return _sharp_chernoff_weighted(s,mu,asrt,w).fun-np.log(alpha)

    LB=s*.5
    while f(LB)>0:
        LB=LB*.5

    return sp.optimize.bisect(f,LB,s)*mx



r'''
      _                                        __ _     _
  ___| | _____   _____ _ __    ___ ___  _ __  / _(_) __| | ___ _ __   ___ ___
 / __| |/ _ \ \ / / _ \ '__|  / __/ _ \| '_ \| |_| |/ _` |/ _ \ '_ \ / __/ _ \
| (__| |  __/\ V /  __/ |    | (_| (_) | | | |  _| | (_| |  __/ | | | (_|  __/
 \___|_|\___| \_/ \___|_|     \___\___/|_| |_|_| |_|\__,_|\___|_| |_|\___\___|

 _       _                       _
(_)_ __ | |_ ___ _ ____   ____ _| |___
| | '_ \| __/ _ \ '__\ \ / / _` | / __|
| | | | | ||  __/ |   \ V / (_| | \__ \
|_|_| |_|\__\___|_|    \_/ \__,_|_|___/

(using the dual problem)

'''

def _h_weighted(s,alpha,asrt,w,t,gam):

    last =np.log(alpha) + s*t

    if last<=0:
        return 0.0

    b=(np.exp(asrt*t) -1) / asrt
    nu = np.clip(np.log(gam*b),0,t*asrt)

    return np.sum(w*(np.exp(nu)-1)/b) + gam*(last-np.sum(w*nu))

def _nustar(s,asrt,last,b,gam,t):
    b=(np.exp(asrt*t) -1) / asrt
    return np.clip(np.log(gam*b),0,t*asrt)

def _gamstar(s,asrt,w,last,b,t):
    def meandiff(gam):
        return np.sum(_nustar(s,asrt,last,b,gam,t)*w) - last

    LB=np.min(1/b)
    UB=np.max(np.exp(t*asrt)/b)

    return sp.optimize.bisect(meandiff,LB,UB)

def confidence_from_inverted_sharp_chernoff(s,a,alpha):
    '''
    Let

        S = sum_i^n X_i
        X_i in [0,a_i]

    Returns mu so that P(S>=s)<=alpha as long as E[S]=mu.
    '''

    _check_arguments(s,a,alpha=alpha)

    ####
    # TODO: if set(a) has one elment, could use
    # confidence_from_hoeffding_thm1(s,n,alim,alpha) instead
    # (would be faster)
    ###

    # keep a in check to maintain numerical stability
    mx=a.max()
    s=s/mx
    a=a/mx
    asrt,w=uniqify(a)

    # if something goes weird, zero is always an acceptable lower bound
    if s<=1e-9: # if s is really small
        return 0.0

    # a good initial choice for t requires that log alpha + st >=0
    t_init = np.max([1,1e-9-np.log(alpha) / s])
    last =np.log(alpha) + s*t_init
    b=(np.exp(asrt*t_init) -1) / asrt
    gam_init = np.clip(_gamstar(s,asrt,w,last,b,t_init),1e-9,np.inf)

    rez=sp.optimize.minimize(
        lambda tgam: -_h_weighted(s,alpha,asrt,w,tgam[0],tgam[1]),
        (t_init,gam_init),
        bounds=[(0,np.inf),(1e-9,np.inf)],
        method='nelder-mead'
    )

    if not rez.success:
        raise Exception("Minimization failed!")

    return -rez.fun*mx

