"""
This script finds the maximum likelihood values of alpha and beta for a
gamma-function prior on the distribution of inverse variances.

The distribution of variances and hence the optimal choice of alpha and beta
depends on the subset of data used.
"""

import numpy as np
from scipy.special import gamma
from scipy.special import digamma
from scipy.optimize import brentq


def prior_params(data, a=0.5, b=0.1, tol=1e-6, maxiter=100):
    """
    finds the maximum likelihood values of alpha and beta for prior
    assuming a gamma-distribution of the inverse variances of the data.

    data:   array containing data. variances are calculated along axis -1
    a:      initial guess for alpha [default 0.5]
    b:      initial guess for beta  [default 0.1]
    tol:    fractional change in log-likelihood after which iterations are
            stopped. Not equivalent to error in alpha, beta which may be much
            larger. [default 1E-8]
    maxiter: Maximum number of iterations before stopping optimization.
            [default 20]
    """
    v = data.var(axis=-1)  # variances of gene expression
    G, n = data.shape  # number of genes, number of samples
    nrh = (n-1)/2  # defined for convenience

    def _LL_g(alpha, beta):
        """ returns array of Log-Likelihoods for every gene """

        return alpha*np.log(beta) - (alpha+nrh)*np.log(beta+n*v/2) \
            + np.log(gamma(alpha+nrh)/gamma(alpha))

    def _LL(alpha, beta):
        """ total log-likelihood function """

        return np.sum(_LL_g(alpha, beta), axis=-1)

    def _beta_optimum(beta, alpha):
        """ returns 0 when beta is optimal given a fixed alpha """
        lhs = np.mean((alpha + nrh)/(beta+n*v/2))
        rhs = alpha/beta
        return lhs - rhs

    def _alpha_optimum(alpha, beta):
        """ returns 0 when alpha is optimal given a fixed beta """

        return np.mean(np.log(1 + (n*v)/(2*beta)) -
                       digamma(alpha+nrh) + digamma(alpha))

    def _update_params(alpha, beta, loops=10):
        """ update alpha and beta """

        for _ in range(loops):
            # find optimal alpha for given beta
            # find initial range in which to look for 0 in alpha_optimum
            alow, ahigh = alpha, alpha
            while True:
                if _alpha_optimum(alow, beta) > 0:
                    ahigh = alow
                    alow = alow/2
                elif _alpha_optimum(ahigh, beta) < 0:
                    alow = ahigh
                    ahigh = ahigh*2
                else:
                    break

            alpha = brentq(_alpha_optimum, alow, ahigh, args=(beta, ))

            # find optimal beta for given alpha
            # find initial range in which to look for 0 in beta_optimum
            blow, bhigh = beta, beta
            while True:
                if _beta_optimum(blow, alpha) > 0:
                    bhigh = blow
                    blow = blow/2
                elif _beta_optimum(bhigh, alpha) < 0:
                    blow = bhigh
                    bhigh = bhigh*2
                else:
                    break

            beta = brentq(_beta_optimum, blow, bhigh, args=(alpha, ))

        return alpha, beta

    i = 0       # count for loops
    try:
        while True:
            i += 1
            print(i)
            LL_old = _LL(a, b)
            a_old, b_old = a, b
            a, b = _update_params(a, b)

            # compute change in LL and exit loop if smaller than tol
            delta_LL = abs((LL_old - _LL(a, b))/LL_old) < tol
            delta_a = abs((a_old - a)/a_old) < tol
            delta_b = abs((b_old - b)/b_old) < tol
            if all([delta_LL, delta_a, delta_b]):
                break

            # raise exception to exit loop after maxiter iterations
            if i >= maxiter:
                raise RuntimeError('No conversion of prior parameters \
                                   for given maxiter')
    except:
        raise

    # simple check that a,b give maximum in log-likelihood within given
    # tolerance
    LL_max = _LL(a, b)
    LL_max = LL_max + abs(LL_max)*tol  # LL may be negative, so LL+tol*|LL|

    error_message = 'Maximization failed. Log-Likelihood higher for {}={}'
    try:
        for k in range(0, 4):
            u = 1 + tol*10**k
            d = 1 - tol*10**k

            if _LL(a*u, b) > LL_max:
                print(_LL(a*u, b), LL_max)
                m = error_message.format('a', a*u)
                raise Exception(m)
            if _LL(a*d, b) > LL_max:
                m = error_message.format('a', a*d)
                raise Exception(m)
            if _LL(a, b*u) > LL_max:
                m = error_message.format('b', b*u)
                raise Exception(m)
            if _LL(a, b*d) > LL_max:
                m = error_message.format('b', b*d)
                raise Exception(m)
    except Exception:
        print('alpha', a)
        print('beta', b)
        raise

    return a, b


def std_eff(x, alpha, beta, axis=0):
    """
    calculate the effective (stabilised) standard deviation of x

    x:      array of measurements
    alpha:  parameter of prior over inverse variance
    beta:   parameter of prior over inverse variance
    axis:   axis along which to calculate std_eff [default=0]
    """
    v = np.var(x, axis=axis)
    n = x.shape[axis]
    return np.sqrt( (v + 2*beta/n) / (1 + 2*alpha/n) )


def main():
    from timeit import default_timer as timer
    # import data as a numpy array
    data_file = './data/expression.tab'
    all_data = np.genfromtxt(data_file, delimiter='\t', skip_header=1)[:, 1:]
    # groups = [[0,1,2,3,4],[5,6,7,8,9]]
    dt = all_data[:, :]
    for tol in [1e-3, 1e-6, 1e-9]:
        start = timer()
        alpha, beta = prior_params(dt, tol=tol, maxiter=200)
        end = timer() - start
        print('tol = {0:.1e}'.format(tol))
        print('elapsed time', end)
        print('alpha = {}'.format(alpha))
        print('beta = {}'.format(beta))


if __name__ == '__main__':
    main()
