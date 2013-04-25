#!/usr/bin/env python
# encoding: utf-8
"""
like.py

Created by vallis on 2012-04-12.
Copyright (c) 2012 Caltech. All rights reserved.
"""

from __future__ import division
import math
import numpy as N, scipy.special as SS, scipy.linalg as SL

from util import timing, numpy_seterr
from constants import *


def Cquad(alphaab,times_f,fH=None):
    if fH is None:
        return N.identity(len(times_f))

    corr = N.zeros((len(times_f),len(times_f)),'d')

    ps, ts = len(alphaab), len(times_f) / len(alphaab)
    for i in range(ps):
        t1, t2 = N.meshgrid(times_f[i*ts:(i+1)*ts],times_f[i*ts:(i+1)*ts])

        # t1, t2 are in days
        x = (2 * math.pi * (day/year) * fH) * (t1 - t2)

        # the correlation function for bandlimited noise with P(f) = A up to f = fH is
        # A fH sin(fH tau) / (fH tau), which has units of [A]/[tau] = T^2
        # but we're interested in normalizing A so that the variance A fH is constant
        # so we drop the fH and join continuously with the no-fH case
        with numpy_seterr(divide='ignore'):
            corr[i*ts:(i+1)*ts,i*ts:(i+1)*ts] = N.where(x==0.0,1.0,N.sin(x)/x)

    return corr

def Cexp(alphaab,times_f,lam,alpha=1.0):
    corr = N.zeros((len(times_f),len(times_f)),'d')

    ps, ts = len(alphaab), len(times_f) / len(alphaab)
    for i in range(ps):
        t1, t2 = N.meshgrid(times_f[i*ts:(i+1)*ts],times_f[i*ts:(i+1)*ts])
        x = (2 * math.pi * (day/year) / lam) * N.abs(t1 - t2)

        corr[i*ts:(i+1)*ts,i*ts:(i+1)*ts] = N.exp(-x**alpha)

    return corr

def Cflat(alphaab,times_f,lam):
    corr = N.zeros((len(times_f),len(times_f)),'d')

    ps, ts = len(alphaab), len(times_f) / len(alphaab)
    for i in range(ps):
        t1, t2 = N.meshgrid(times_f[i*ts:(i+1)*ts],times_f[i*ts:(i+1)*ts])
        x = N.abs(t1 - t2)

        # corr[i*ts:(i+1)*ts,i*ts:(i+1)*ts] = N.where(x < lam,1.0,0.0)
        corr[i*ts:(i+1)*ts,i*ts:(i+1)*ts] = N.where(x < lam,1.0 - x/lam,0.0)

    return corr

def Cbandlim(alphaab,times_f,fL,fH):
    corr = [N.zeros((len(times_f),len(times_f)),'d') for j in range(2)]

    ps, ts = len(alphaab), len(times_f) / len(alphaab)
    for i in range(ps):
        t1, t2 = N.meshgrid(times_f[i*ts:(i+1)*ts],times_f[i*ts:(i+1)*ts])
        deltat = t1 - t2

        x = (2 * math.pi * (day/year) * fL) * deltat
        with numpy_seterr(divide='ignore'):
            corr[0][i*ts:(i+1)*ts,i*ts:(i+1)*ts] = N.where(x==0.0,1.0,N.sin(x)/x)

        x = (2 * math.pi * (day/year) * fH) * deltat
        with numpy_seterr(divide='ignore'):
            corr[1][i*ts:(i+1)*ts,i*ts:(i+1)*ts] = N.where(x==0.0,1.0,N.sin(x)/x)

    return corr

def Cbands(alphaab,times_f,fH=12.0,bands=4):
    deltaf = fH/bands
    corr = [N.zeros((len(times_f),len(times_f)),'d') for j in range(bands)]

    ps, ts = len(alphaab), len(times_f) / len(alphaab)
    for i in range(ps):
        t1, t2 = N.meshgrid(times_f[i*ts:(i+1)*ts],times_f[i*ts:(i+1)*ts])
        deltat = t1 - t2

        for j in range(bands):
            x = (2 * math.pi * (day/year) * (j + 1) * deltaf) * deltat

            with numpy_seterr(divide='ignore'):
                corr[j][i*ts:(i+1)*ts,i*ts:(i+1)*ts] = N.where(x==0.0,1.0,N.sin(x)/x)

    return corr

def Cpn_efac(alphaab,times_f,cpn,efac):
    ps, ts = len(alphaab), len(times_f) / len(alphaab)

    if not isinstance(efac,(list,tuple,N.ndarray)):
        efac = [efac] * ps

    cpndiag = N.diag(cpn)

    for i in range(ps):
        cpndiag[i*ts:(i+1)*ts] *= efac[i]

    return N.diag(cpndiag)


# watch different definitions for the exponents and amplitude
def Cred_100ns(alphaab,times_f,A=5.77e-22,alpha=1.7,fL=1.0/10000):
    corr = N.zeros((len(times_f),len(times_f)),'d')

    ps, ts = len(alphaab), len(times_f) / len(alphaab)

    if not isinstance(A,(list,tuple,N.ndarray)):
        A = [A] * ps
    if not isinstance(alpha,(list,tuple,N.ndarray)):
        alpha = [alpha] * ps

    for i in range(ps):
        t1, t2 = N.meshgrid(times_f[i*ts:(i+1)*ts],times_f[i*ts:(i+1)*ts])

        # t1, t2 are in days
        x = 2 * math.pi * (day/year) * fL * N.abs(t1 - t2)

        if abs(alpha[i] - 2.0) < 1e-7:
            power = (-math.pi/2 + (math.pi/2 - math.pi*EulerGamma/2) * (alpha[i] - 2.0)) * x**(alpha[i] - 1)
        else:
            power = SS.gamma(1-alpha[i]) * math.sin(0.5*math.pi*alpha[i]) * x**(alpha[i] - 1)

        # HypergeometricPFQ[{1/2-alpha/2},{1/2,3/2-alpha/2},-(1/4)fL^2 Tau^2]/(alpha - 1)
        ksum = SS.hyp1f2(0.5 - 0.5*alpha[i],0.5,1.5-0.5*alpha[i],-0.25*x**2)[0] / (alpha[i] - 1)

        # this must be multiplied with an amplitude given in units of s^(3-alpha),
        # so it must have units of s^(alpha - 1) to yield s^2;
        # we further multiply by 1e14 for units of 100 ns
        corr[i*ts:(i+1)*ts,i*ts:(i+1)*ts] = (A[i] * 1.0e14 * year**(alpha[i] - 1) * fL**(1-alpha[i])) * (power + ksum)

    return corr

def Cgw_dm_year(alphaab,times_f,freqs_f,d1000=1.0,gamma=8.0/3.0,freq_ref=1e-6*clight/0.20,fL=1.0/500,fH=None):
    # eq. (12) from Keith et al.
    # d1000 (@ 1000 days) is given in us^2, must be converted to s^2
    # spectrum given in yr^3
    # timing-fluctuation P(f) = A2 (f yr)^(-8/3) yr^3, with A2 given by

    A2 = 0.0112 * (1e-12 * d1000) * (1000.0 * day)**(-5.0/3.0) * year**(-1.0/3.0)

    cgw = Cgw_reg_year(alphaab,times_f,alpha=1.5 - 0.5*gamma,fL=fL,fH=fH,decompose=False)

    # lambda_ref is given in meters, freq_ref in MHz like the fs
    # TO DO: this matrix is constant, so we could cache it...
    f1, f2 = N.meshgrid(freqs_f,freqs_f)
    freqnorm = (freq_ref/f1)**2 * (freq_ref/f2)**2

    return A2 * (freqnorm * cgw)

def Cgw_reg_year(alphaab,times_f,alpha=-2/3,fL=1.0/500,fH=None,decompose=False):
    t1, t2 = N.meshgrid(times_f,times_f)

    x = 2 * math.pi * (day/year) * fL * N.abs(t1 - t2)

    # print N.min(x), N.max(x), N.max(t2 - t1)
    year100ns = 1.0   # was year100ns = year/1e-7 for Ggw_reg_year

    norm = (year100ns**2 * fL**(2*alpha - 2)) * 2**(alpha - 3) / (3 * math.pi**1.5 * SS.gamma(1.5 - alpha))

    if fH is not None:
        # introduce a high-frequency cutoff
        xi = fH/fL

        # avoid the gamma singularity at alpha = 1
        if abs(alpha - 1) < 1e-6:
            diag = math.log(xi) + (EulerGamma + math.log(0.5 * xi)) * math.log(xi) * (alpha - 1)
        else:
            diag = 2**(-alpha) * SS.gamma(1 - alpha) * (1 - xi**(2*alpha - 2))

        with numpy_seterr(divide='ignore'):
            if decompose:
                corr = N.where(x==0,0.0,x**(1 - alpha) * (SS.kv(1 - alpha,x) - xi**(alpha - 1) * SS.kv(1 - alpha,xi * x)) - diag)
            else:
                corr = N.where(x==0,norm * diag,norm * x**(1 - alpha) * (SS.kv(1 - alpha,x) - xi**(alpha - 1) * SS.kv(1 - alpha,xi * x)))
    else:
        if decompose:
            diag = 2**(-alpha) * SS.gamma(1 - alpha)
            corr = N.where(x==0,0,x**(1 - alpha) * SS.kv(1 - alpha,x) - diag)
        else:
            # testing for zero is dangerous, but kv seems to behave OK for arbitrarily small arguments
            corr = N.where(x==0,norm * 2**(-alpha) * SS.gamma(1 - alpha),
                                norm * x**(1 - alpha) * SS.kv(1 - alpha,x))

    ps, ts = len(alphaab), len(times_f) / len(alphaab)
    for i in range(ps):
        for j in range(ps):
            corr[i*ts:(i+1)*ts,j*ts:(j+1)*ts] *= alphaab[i,j]

    if decompose:
        return norm, diag, corr
    else:
        return corr


def Cgw_100ns(alphaab,times_f,alpha=-2/3,fL=1.0/10000,approx_ksum=False):
    """Compute the residual covariance matrix for an hc = 1 x (f year)^alpha GW background.
    Result is in units of (100 ns)^2."""

    t1, t2 = N.meshgrid(times_f,times_f)

    # t1, t2 are in units of days; fL in units of 1/year (sidereal for both?)
    # so typical values here are 10^-6 to 10^-3
    x = 2 * math.pi * (day/year) * fL * N.abs(t1 - t2)

    # note that the gamma is singular for all half-integer alpha < 1.5
    #
    # for -1 < alpha < 0, the x exponent ranges from 4 to 2 (it's 3.33 for alpha = -2/3)
    # so for the lower alpha values it will be comparable in value to the x**2 term of ksum
    #
    # possible alpha limits for a search could be [-0.95,-0.55] in which case the sign of `power`
    # is always positive, and the x exponent ranges from ~ 3 to 4... no problem with cancellation

    # the variance, as computed below, is in units of year^2; we convert it
    # to units of (100 ns)^2, so the multiplier is ~ 10^28
    year100ns = year/1e-7
    tol = 1e-5

    # the exact solutions for alpha = 0, -1 should be acceptable in a small interval around them...
    if abs(alpha) < 1e-7:
        cosx, sinx = N.cos(x), N.sin(x)

        power = cosx - x * sinx
        sinint, cosint = SL.sici(x)

        corr = (year100ns**2 * fL**-2) / (24 * math.pi**2) * (power + x**2 * cosint)
    elif abs(alpha + 1) < 1e-7:
        cosx, sinx = N.cos(x), N.sin(x)

        power = 6 * cosx - 2 * x * sinx - x**2 * cosx + x**3 * sinx
        sinint, cosint = SL.sici(x)

        corr = (year100ns**2 * fL**-4) / (288 * math.pi**2) * (power - x**4 * cosint)
    else:
        # leading-order expansion of Gamma[-2+2*alpha]*Cos[Pi*alpha] around -0.5 and 0.5
        if   abs(alpha - 0.5) < tol:
            cf =  math.pi/2   + (math.pi - math.pi*EulerGamma)              * (alpha - 0.5)
        elif abs(alpha + 0.5) < tol:
            cf = -math.pi/12  + (-11*math.pi/36 + EulerGamma*math.pi/6)     * (alpha + 0.5)
        elif abs(alpha + 1.5) < tol:
            cf =  math.pi/240 + (137*math.pi/7200 - EulerGamma*math.pi/120) * (alpha + 1.5)
        else:
            cf = SS.gamma(-2+2*alpha) * math.cos(math.pi*alpha)

        power = cf * x**(2-2*alpha)

        # Mathematica solves Sum[(-1)^n x^(2 n)/((2 n)! (2 n + 2 alpha - 2)), {n, 0, Infinity}]
        # as HypergeometricPFQ[{-1+alpha}, {1/2,alpha}, -(x^2/4)]/(2 alpha - 2)
        # the corresponding scipy.special function is hyp1f2 (which returns value and error)
        # TO DO, for speed: could replace with the first few terms of the sum!
        if approx_ksum:
            ksum = 1.0 / (2*alpha - 2) - x**2 / (4*alpha) + x**4 / (24 * (2 + 2*alpha))
        else:
            ksum = SS.hyp1f2(alpha-1,0.5,alpha,-0.25*x**2)[0]/(2*alpha-2)

        # this form follows from Eq. (A31) of Lee, Jenet, and Price ApJ 684:1304 (2008)

        corr = -(year100ns**2 * fL**(-2+2*alpha)) / (12 * math.pi**2) * (power + ksum)

    # multiply by alphaab; there must be a more numpythonic way to do it
    ps, ts = len(alphaab), len(times_f) / len(alphaab)
    for i in range(ps):
        for j in range(ps):
            corr[i*ts:(i+1)*ts,j*ts:(j+1)*ts] *= alphaab[i,j]

    return corr


# compute C in "natural" units of days^2
def Cgw_days(alphaab,times_f,alpha=-2/3,fL=1.0/10000,approx_ksum=False):
    """Compute the residual covariance matrix for an hc = 1 x (f year)^alpha GW background.
    Result is in units of (days)^2."""

    t1, t2 = N.meshgrid(times_f,times_f)

    # t1, t2 are in units of days
    x = 2 * math.pi * N.abs(t1 - t2)
    power = (day/year)**(-2*alpha) * (SS.gamma(-2+2*alpha) * math.cos(math.pi*alpha)) * x**(2-2*alpha)

    # Mathematica solves Sum[(-1)^n x^(2 n)/((2 n)! (2 n + 2 alpha - 2)), {n, 0, Infinity}]
    # as HypergeometricPFQ[{-1+alpha}, {1/2,alpha}, -(x^2/4)]/(2 alpha - 2)
    # the corresponding scipy.special function is hyp1f2 (which returns value and error)
    # TO DO, for speed: could replace with the first few terms of the sum!
    # ksum = SS.hyp1f2(alpha-1,0.5,alpha,-0.25*x**2)[0]/(2*alpha-2)

    if not fL:                  # do not include any low-fcut correction
        ksum = 0
    else:
        fLd = (day/year) * fL

        if approx_ksum:
            ksum = (year/day)**2 * fL**(-2+2*alpha) * (1.0 / (2*alpha - 2) - (fLd * x)**2 / (4*alpha) + (fLd * x)**4 / (24 * (2 + 2*alpha)))
        else:
            ksum = (year/day)**2 * fL**(-2+2*alpha) * SS.hyp1f2(alpha-1,0.5,alpha,-0.25*(fLd * x)**2)[0]/(2*alpha-2)

    corr = -(power + ksum) / (12 * math.pi**2)

    # multiply by alphaab; there must be a more numpythonic way to do it
    ps, ts = len(alphaab), len(times_f) / len(alphaab)
    for i in range(ps):
        for j in range(ps):
            corr[i*ts:(i+1)*ts,j*ts:(j+1)*ts] *= alphaab[i,j]

    return corr



def Cpn(error_f,noise='white'):
    if noise != 'white':
        raise NotImplementedError

    return N.diag(error_f**2)


# test my expression for the likelihood on a simple model
#
# OK, this works as expected... from theory, the ML estimator for the (non-squared) amplitude multiplier
# of the covariance matrix follows a chi(n) distribution (with n the data points) normalized by sqrt(n), times the true value.
# This means that for a reasonable number of data points, the estimator is unbiased; furthermore, the variance tends to 0.5 / n
# (times the true value squared). So our IPTA simulation, without noise, _should_ be able to recover the right value.
#
# These statistics depend on the identity D . C^-1 . D^T = I, where C = D^T D and random data = D^T . normal vector;
# this is ostensibly true (at small dimension) for the scipy cholesky functions.

def testcholesky():
    A = N.array([[4,1],[1,2]],'d')

    x = N.dot(SL.cholesky(A).T,N.random.randn(2))

    As = N.linspace(0.2,4,50)
    res = []

    for a in As:
        cf = SL.cho_factor(a**2 * A)
        logl = -0.5 * N.dot(x,SL.cho_solve(cf,x)) - 0.5 * len(x) * math.log(2*math.pi) - 0.5 * N.sum(N.log(N.diag(cf[0])**2))

        res.append((a,logl))

    return N.array(res)



# possibilities for speeding up the inversion, as discussed with Loris:
#
# - basically solve the linear equation x = C.y and compute x.y
# - if determinant needed Cholesky decomposition
#   - in numpy dot(x,cho_solve(cho_factor(C), x))
#   - determinant given by prod(diag(cho_factor(C)))
# - if determinant not needed, gradient descent
# - try Karhunen–Loève theorem
# - see if matrix can be expressed in terms of objects with lower rank
# - do SVD, analyze structure to drop some singular values, work with projectors


def Gproj(times_f,npulsars):
    # M is the matrix such that M.xi gives the deterministic contribution to the residuals
    # with xi the coefficients of dt = sum_i xi_i f_i(t); in our case i = 1--3
    # if we order xi_{ai} as (xi_{11} xi_{12} xi_{13} xi_{21} xi_{22} xi_{23} ...)^T
    # then M is |1 t_{11} t_{11}^2 0 0      0        ... |
    #           |1 t_{12} t_{12}^2 0 0      0            |
    #           |       ...             ...              |
    #           |1 t_{1n} t_{1n}^2 0 0      0        ... |
    #           |0 0      0        1 t_{21} t_{21}^2     |
    #           |0 0      0        1 t_{22} t_{22}^2 ... |
    #           |       ...               ...
    #           |0 0      0        1 t_{2n} t_{2n}^2 ... |
    #           |       ...               ...            |

    ntimes = len(times_f) / npulsars
    M = N.zeros((npulsars*ntimes,npulsars*3),'d')

    # is there some smart numpy notation for block-diagonal matrices?
    time2_f = times_f**2
    for i in range(npulsars):
        M[i*ntimes:(i+1)*ntimes,i*3]   = 1
        M[i*ntimes:(i+1)*ntimes,i*3+1] = times_f[i*ntimes:(i+1)*ntimes]
        M[i*ntimes:(i+1)*ntimes,i*3+2] = time2_f[i*ntimes:(i+1)*ntimes]

    U, s, Vh = SL.svd(M)

    # n = npulsars * ntimes, n - m = npulsars*(ntimes-3)
    return U[:,npulsars*3:].copy()     # return contiguous


# get the design matrix from the tempo2 output (with Rutger's designmatrix plugin)
def Gdesi(design,npulsars):
    U, s, Vh = SL.svd(design)

    return U[:,design.shape[1]:].copy()

def Gdesi2(design,meta):
    npulsars, times = len(meta), len(design)
    ptimes = times / npulsars
    pars = [line['pars'] for line in meta]

    U = N.zeros((times,times - sum(pars)),'d')

    j,k = 0,0
    for i in range(npulsars):
        u, s, v = SL.svd(design[i*ptimes:(i+1)*ptimes,j:j+pars[i]])
        U[i*ptimes:(i+1)*ptimes,k:k+(ptimes - pars[i])] = u[:,pars[i]:]

        j = j + pars[i]
        k = k + ptimes - pars[i]

    return U

def simulate(alphaab,times_f,cgw,cpn,A=5e-14,n=1):
    # assumes a diagonal cpn...
    noise = n * N.sqrt(N.diag(cpn)) * N.random.randn(len(times_f))
    sgwb = A * N.dot(SL.cholesky(cgw).T,N.random.randn(len(times_f)))

    return sgwb + noise


def logL(resid,cgw,cpn,A=5e-14,n=1,cgwunit='100ns'):
    if cgwunit == '100ns':
        # cgw and cpn already have the same units
        C = A**2 * cgw + n**2 * cpn
    elif cgwunit == 'days':
        # take cgw in days^2; we wish to convert it to (100 ns)^2, appropriate for dominant noise...
        C = (A**2 * (day/1e-7)**2) * cgw + cpn
    else:
        raise NotImplementedError

    # we rewrite -0.5 * math.log(N.prod(N.diag(cf[0])**2)) as -0.5 * N.sum(N.log(N.diag(cf[0])**2))
    cf = SL.cho_factor(C)
    res = -0.5 * N.dot(resid,SL.cho_solve(cf,resid)) - 0.5 * len(resid) * math.log((2*math.pi)) - 0.5 * N.sum(N.log(N.diag(cf[0])**2))

    # alternative expression, explicitly in terms of Cholesky decomposition matrix:
    # cf = SL.cholesky(C)
    # res = -0.5 * N.dot(resid,SL.solve(cf,SL.solve(cf.T,resid))) - 0.5 * len(resid) * math.log((2*math.pi)) - 0.5 * N.sum(N.log(N.diag(cf)**2))

    # alternative expression, in terms of matrix inverse (determinant may blow up though)
    # res = -0.5 * N.dot(resid,N.dot(N.linalg.inv(C),resid)) - 0.5 * len(resid) * math.log((2*math.pi)) - 0.5 * math.log(N.linalg.det(C))

    return res


try:
    from matmul import matmul, matmul_view
    blas = True
except ImportError:
    blas = False

blas = False

# run doctests with python -m doctest background.py
def blockmul(A,B,meta):
    """Computes B.T . A . B, where B is a block-diagonal design matrix
    with block heights m = len(A) / len(meta) and block widths m - meta[i]['pars'].
    Calls matmul.pyx to access BLAS dgemm function.

    >>> a = N.random.randn(8,8)
    >>> a = a + a.T
    >>> b = N.zeros((8,5),'d')
    >>> b[0:4,0:2] = N.random.randn(4,2)
    >>> b[4:8,2:5] = N.random.randn(4,3)
    >>> meta = [{'pars': 2},{'pars': 1}]
    >>> c = blockmul(a,b,meta) - N.dot(b.T,N.dot(a,b))
    >>> N.max(N.abs(c))
    0.0
    """

    n,p = A.shape[0], B.shape[1]    # A is n x n, B is n x p

    if (A.shape[0] != A.shape[1]) or (A.shape[1] != B.shape[0]):
        raise ValueError('incompatible matrix sizes')

    res1 = N.zeros((n,p),'d')
    res2 = N.zeros((p,p),'d')

    npulsars = len(meta)
    m = n/npulsars          # times (assumed the same for every pulsar)

    psum = 0
    for i in range(npulsars):
        # each A matrix is n x m, with starting column index = i * m
        # each B matrix is m x (m - p_i), with starting row = i * m, starting column s = sum_{k=0}^{i-1} (m - p_i)
        # so the logical C dimension is n x (m - p_i), and it goes to res1[:,s:(s + m - p_i)]

        pi = m - meta[i]['pars']
        if blas:
            matmul_view(A,(n,m),(0,i*m),B,(m,pi),(i*m,psum),res1,(0,psum),transpose_A=False)
        else:
            res1[:,psum:psum+pi] = N.dot(A[:,i*m:(i+1)*m],B[i*m:(i+1)*m,psum:psum+pi])
        psum = psum + pi

    psum = 0
    for i in range(0,npulsars):
        pi = m - meta[i]['pars']
        if blas:
            matmul_view(B,(pi,m),(psum,i*m),res1,(m,p),(i*m,0),res2,(psum,0),transpose_A=True)
        else:
            res2[psum:psum+pi,:] = N.dot(B.T[psum:psum+pi,i*m:(i+1)*m],res1[i*m:(i+1)*m,:])
        psum = psum + pi

    return res2


def logL2(resid,alphaab,times_f,gmat,meta,cpn,A=5e-14,alpha=-2.0/3.0,Ared=None,alphared=None,efac=None):
    with numpy_seterr(all='raise'):
        try:
            cgw = A**2 * Cgw_100ns(alphaab,times_f,alpha,fL=1.0/500)
        except FloatingPointError:
            print "Hmm... problem at A = %s, alpha = %s" % (A,alpha)
            raise

    if Ared is not None:
        cgw = cgw + Cred_100ns(alphaab,times_f,A=Ared,alpha=alphared,fL=1.0/500)

    if efac is not None:
        C = blockmul(cgw + Cpn_efac(alphaab,times_f,cpn,efac),gmat,meta)
    else:
        C = blockmul(cgw + cpn,gmat,meta)

    try:
        cf = SL.cho_factor(C)
        res = -0.5 * N.dot(resid,SL.cho_solve(cf,resid)) - 0.5 * len(resid) * math.log((2*math.pi)) - 0.5 * N.sum(N.log(N.diag(cf[0])**2))
    except N.linalg.LinAlgError:
        print "Problem inverting matrix at A = %s, alpha = %s; also Ared, alphared:" % (A,alpha)
        print Ared
        print alphared

        raise

    return res



# for this computation, see Eq. (A30) of Lee, Jenet, and Price ApJ 684:1304 (2008)
def alphamat(meta):
    k = [(math.cos(p['dec'])*math.cos(p['ra']),
          math.cos(p['dec'])*math.sin(p['ra']),
          math.sin(p['dec'])) for p in meta]

    costh = N.array([[N.dot(k1,k2) for k2 in k] for k1 in k])
    sth22 = 0.5 * (1 - costh)

    with numpy_seterr(all='ignore'):
        res = 1.5 * sth22 * N.log(sth22)
        N.fill_diagonal(res,0)

    return res - 0.25 * sth22 + 0.5 + 0.5 * N.diag(N.ones(len(k)))
