"""
For two or more given groups of gene expression samples this script gives an
ordered list of the most promising marker genes that would differentiate between the two
groups.

The scoring is based on the conditional entropy of the classes given the data.
"""
# import relevant python libraries
import numpy as np
from scipy.special import gamma
from scipy.special import digamma
from scipy.special import lambertw
import scipy.integrate as integrate
from .prior_gamma import prior_params


def create_ranges(start, stop, N):
    steps = (1.0/(N-1)) * (stop - start)
    return steps[:, None]*np.arange(N) + start[:, None]


def entropy(data, all_c, ncond=None, k=6, N=1E4):
    """ calculates the conditional entropy H(c|x) of multiple genes
    data:   NxM array of N genes across M conditions
    all_c:  list of list of indices indicating which classes the samples are
            split into
    ncond:  number of conditions in each group; gives prior probability
            that a sample is in a group. [default = None, uniform prior]
    k:      parameter setting size of integration domain for H(x)
            [default = 6]
    N:      parameter setting number of bins in integration for H(x)
            [default = 1E4]
    """
    # make sure data are in form of 2D array
    data = np.asarray(data)
    dim = data.ndim
    if dim == 1:
        data = data.reshape(1, -1)
    elif dim == 2:
        pass
    else:
        message = 'dimension of data is {}, but should be 1 or 2'.format(dim)
        raise Exception(message)

    N_g = data.shape[0]  # number of genes
    H = np.zeros(N_g)  # initialize entropy array

    N_c = len(all_c)  # number of classes

    if ncond is None:
        ncond = np.ones(N_c)
    else:
        ncond = np.asarray(ncond)
    prior_c = ncond/np.sum(ncond)

    # find alpha and beta values
    abparams = [prior_params(data[:, c]) for c in all_c]
    all_alpha = np.array([a[0] for a in abparams])
    all_beta = np.array([a[1] for a in abparams])

    for c, alphac, betac, pc in zip(all_c, all_alpha, all_beta, prior_c):
        datac = data[:, c]
        varc = datac.var(axis=-1)
        nc = len(c)  # size of class c
        Vc = (nc + 1)*(varc + 2*betac/nc)
        gc = alphac + (nc - 1)/2
        Zc = gamma(gc) / (np.sqrt(Vc*np.pi) * gamma(gc-0.5))

        H_xc = -np.log(Zc) + gc*(digamma(gc) - digamma(gc - 0.5))  # H(x|c)
        H += pc*(H_xc - np.log(pc))

        def P(xx, cc, alphacc, betacc):
            datacc = data[:, cc]
            varcc = datacc.var(axis=-1)
            ncc = len(cc)  # size of class c
            Vcc = (ncc + 1)*(varcc + 2*betacc/ncc)
            xcc = datacc.mean(axis=-1)
            gcc = alphacc + (ncc - 1)/2
            Zcc = gamma(gcc) / (np.sqrt(Vcc*np.pi) * gamma(gcc-0.5))
            res = Zcc[:, None] * \
                np.power((1 + (xx - xcc[:, None])**2 / Vcc[:, None]), -gcc)
            return res

        def integrand(xx):
            Psum = np.sum(prior_c[:, None, None] *
                          np.array([P(xx, cc, a, b) for cc, a, b
                                    in zip(all_c, all_alpha, all_beta)]),
                          axis=0)
            return - P(xx, c, alphac, betac)*np.log(Psum)

        def integrate_simple():
            xd = np.sqrt(Vc * (-1 + 10**(k/gc)))
            xc = datac.mean(axis=-1)
            xmin = xc - xd
            xmax = xc + xd

            x = create_ranges(xmin, xmax, N)
            P_vals = integrand(x)
            integral = integrate.trapz(P_vals, x, axis=1)
            return integral

        H_x = integrate_simple()
        H -= pc*H_x

    return H  # /N_c


def find_error_probability(H):
    return np.real_if_close(-H / lambertw(-H/np.e, k=-1))


def entropyfunc(p):
    return - (p*np.log(p) + (1-p)*np.log(1-p))
