import torch

import numpy as np
import scipy.special as sc

from scipy.optimize import brentq

def generate_so3_sampling_grid(n_beta=8, n_alpha=8, n_gamma=8):
    """
    @param:  (n_beta, n_alpha,  n_gamma) uniform grid sizes. 
    @return: np.ndarray (n_beta x n_alpha  x n_gamma, 3)

    @note: Integration on beta use quarature rule?
    """
    
    _beta  = np.linspace(start=0., stop=  np.pi, num=n_beta,  endpoint=False, dtype=float)
    _alpha = np.linspace(start=-np.pi, stop=np.pi, num=n_alpha, endpoint=False, dtype=float)
    _gamma = np.linspace(start=-np.pi, stop=np.pi, num=n_gamma, endpoint=False, dtype=float)

    _beta_c, _alpha_c, _gamma_c = np.meshgrid(_beta, _alpha, _gamma, sparse=False, indexing="ij") 
    _beta_f, _alpha_f, _gamma_f =  _beta_c.flatten(), _alpha_c.flatten(), _gamma_c.flatten() 

    sample_grid = np.stack((_beta_f, _alpha_f,  _gamma_f), axis=1) 

    # The weight for Riemann Summation.
    sample_weight = torch.sin(torch.tensor(_beta_f, dtype=torch.float))

    return sample_grid, sample_weight


def spherical_bessel_roots(n=0, m=5):
    """
    @param  n: order of spherical Bessel
    @param  m: number of roots.
    @return first m zeros of SphericalBesselJ[k, z] for 0\le k \le n, 0 is excluded. 
    """
    _zeros     = np.zeros((n+1, m))
    _zeros[0]  = np.arange(1, m + 1) * np.pi
    _pts       = np.arange(1, m + n + 1) * np.pi
    _roots     = np.zeros(m + n)

    for i in range(1, n+1):
        for j in range(m + n - i):
            _root_rec = brentq(lambda x: sc.spherical_jn(i, x), \
                                _pts[j], _pts[j+1])
            _roots[j] = _root_rec 
        _pts = _roots 
        _zeros[i][:m] = _roots[:m]
    
    return _zeros

def spherical_bessel_basis(n, root,  z=0, normalization=False):
    """
    @param n             : order of spherical Bessel
    @param root          : a root of n-th spherical Bessel.
    @param z             : evaluation at z.
    @param normalization : normalization. It is not necessary in the computation.
    @return evaluation of SphericalBesselJ[n, root * z] if not normalized.
    """
    if normalization:
        return np.sqrt(2) * sc.spherical_jn(n, root * z) / np.abs( sc.spherical_jn(n+1, root) )
    else:
        return sc.spherical_jn(n, root * z) 

def cartesian_spherical(x,y,z):
    """
    @param x, y, z  : coordinates
    @return azimuth angle [0, 2*pi] 
            polar angle [0, pi]
            r \ge 0
    """
    azimuth = np.arctan2(y,x)
    rho2 = x**2 + y**2
    polar = np.arctan2(np.sqrt(rho2),z)
    r = np.sqrt(rho2 + z**2)
    return azimuth, polar, r

def spherical_harmonics(m , n, theta, phi):
    """
    @param m:     order of spherical harmonics.
    @param n:     degree |m| \le n.
    @param theta: azimuth angle, [0, 2* pi].
    @param phi:   polar angle,   [0, pi].

    @return       complex float.
    """
    return sc.sph_harm (m, n, theta, phi)

def wiger_d_func(l, n, m, theta):
    if n == 0 and m == 0:
        return sc.eval_legendre(np.float(l), np.cos(theta))

    if np.abs(n) > abs(m):
        if n > 0:
            lp, a, b, c = n, n-m, n+m, n-m 
        else:
            lp, a, b, c = -n, m-n, -n-m, 0
    else:
        if m > 0:
            lp, a, b, c = m, m -n, n +m, 0
        else:
            lp, a, b, c = -m, n-m, -n-m, n-m

    u = np.sin(0.5 * theta)
    v = np.cos(0.5 * theta)
    x = v * v - u * u

    d0 = (1 - 2 * (c&1) ) * np.sqrt(sc.binom(np.float(a + b), np.float(a) )) * np.power(u, a) * np.power(v , b) 

    d1 = 0
    j  = n * m

    if l < lp:
        return 0.
    elif l == lp:
        return d0
    else:
        for l_ in range(lp+1, l+1):
            u = (1.0 - 1.0/(l_ - n)) * (1.0 - 1.0/(l_ +n))
            v = (1.0 - 1.0/(l_ - m)) * (1.0 - 1.0/(l_ +m))

            d2 = d1 
            d1 = d0

            d0 = (l_ * x - j / (l_ -1)) * np.sqrt((1-u)*(1-v)) * d1 - (1.0 + 1.0/(l_ -1)) * (np.sqrt(u * v)) * d2

        return d0


def wignerD(b, j, n, beta, alpha, gamma):
    """
    @param D, j, n, beta, alpha, gamma
    @return complex valued D_b^{jn}(beta, alpha,  gamma).
    """
    val = wiger_d_func(b, j, n, beta) * np.exp(-1j * (j * alpha + n * gamma)) 
    return val.real, val.imag
