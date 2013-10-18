#!/usr/bin/env python
# encoding: utf-8
"""
sampleutils.py

Created by vallis on 2013-04-02.
Copyright (c) 2013 California Institute of Technology
"""

from __future__ import division
import os, json, re
import numpy as N
import matplotlib.pyplot as P

# --- get pulsar data from pickles ---

def getpickle(picklefile):
    import cPickle

    return cPickle.load(file(picklefile,'r'))

def savepickle(pulsardata,picklefile):
    import cPickle

    cPickle.dump(pulsardata,file(picklefile,'w'))


# --- get pulsar data from tempo2 files ---

def findtempo2(pulsarfile,pulsardir='.',timsuffix='',debug=True):
    """Locate tempo2 par and tim files. If pulsardir is not given,
    try a few notable locations. Optionally, supplement the tim-file suffix."""

    # locate the tempo2 files: try some suggested directories

    if '.par' in pulsarfile or '.tim' in pulsarfile:
        pulsarfile = pulsarfile[:-4]

    if timsuffix and timsuffix[0] != '.':
        timsuffix = '.' + timsuffix

    for parfile,timfile in [('{0}/{1}.par'.format(pulsardir,pulsarfile),                  '{0}/{1}.tim{2}'.format(pulsardir,pulsarfile,timsuffix)),
                            ('../nanograv/par/{0}_NANOGrav_dfg+12.par'.format(pulsarfile),'../nanograv/tim/{0}_NANOGrav_dfg+12.tim{1}'.format(pulsarfile,timsuffix)),
                            ('../eptadata/par/{0}.par'.format(pulsarfile),'../eptadata/tim/{0}.tim{1}'.format(pulsarfile,timsuffix)),
                            ('../epta/{0}.par'.format(pulsarfile),                        '../epta/{0}.tim{1}'.format(pulsarfile,timsuffix)),
                            ('../tempo2/open1/{0}.par'.format(pulsarfile),                '../tempo2/open1/{0}.tim{1}'.format(pulsarfile,timsuffix))                 ]:
        if os.path.isfile(parfile) and os.path.isfile(timfile):
            if debug:
                print "Using files %s and %s." % (parfile,timfile)
            return pulsarfile, parfile, timfile

    raise OSError, "Cannot find par and tim files for pulsar {0}".format(pulsarfile)


def gettempo2(pulsarfile,pulsardir='.',timsuffix='',summary=None,debug=False):
    """Load a pickled archive, or use libstempo.pyx to load pulsar data from tempo2 par and tim files."""

    pfile = 'pickles/{0}.pickle'.format(pulsarfile)

    if os.path.isfile(pfile):
        if debug:
            print "Using pickled file %s." % pfile
        return pulsarfile, getpickle(pfile)

    pulsarfile, parfile, timfile = findtempo2(pulsarfile,pulsardir=pulsardir,timsuffix=timsuffix,debug=debug)

    import libstempo as T

    pulsar = T.tempopulsar(parfile,timfile)

    meta = N.zeros((1,),dtype=[('name','a32'),('ra','f8'),('dec','f8'),('designpars','i4'),('pars','i4')])

    meta['name']       = pulsarfile
    meta['ra']         = pulsar['RAJ'].val
    meta['dec']        = pulsar['DECJ'].val
    meta['designpars'] = 0                  # really an offset
    meta['pars']       = len(pulsar.pars)

    # get data (will make copies)

    times_f = N.array(pulsar.toas() - pulsar['PEPOCH'].val,'d')   # days
    resid_f = N.array(pulsar.residuals(),'d')                     # seconds
    error_f = N.array(1e-6 * pulsar.toaerrs,'d')                  # seconds
    freqs_f = N.array(pulsar.freqs,'d')                           # MHz

    # get design matrix

    desi = pulsar.designmatrix()

    if desi.shape[1] != meta['pars'] + 1:
        msg = 'The number of fitting parameters ({0}) and the size of design matrix ({1}x{2}) do not match!'.format(
            meta['pars'],desi.shape[0],desi.shape[1])
        raise ValueError, msg

    # outlier-removal logic
    # argout = lambda data,m=2: N.abs(data - N.mean(data)) < m * N.std(data)
    # kx = argout(resid_f,3)
    # print "Removing {0} TOAs".format(desi.shape[0] - N.sum(kx))
    # desi, times_f, resid_f, error_f = desi[kx,:], times_f[kx], resid_f[kx], error_f[kx]

    # rescale residuals and errors to units of 100 ns (yes, hardcoded)

    resid_f = resid_f / 1e-7
    error_f = error_f / 1e-7

    if summary:
        summary['parfile'] = parfile
        summary['timfile'] = timfile

    return pulsarfile,(meta,desi,times_f,resid_f,error_f,freqs_f)


# legacy list of model parameters

parlists = {
    'walk':         ['Anu','Anudot','log10_efac','log10_equad'],
    'efac':         ['log10_efac'],
    'white':        ['log10_efac','log10_equad'],
    'whitec':       ['log10_efac','log10_equad'],
    'powerlaw':     ['Ared','gammared','log10_efac','log10_equad'],
    'powerlawc':    ['Ared','gammared','log10_efac','log10_equad'],
    'powerlawwc':   ['Ared','gammared','log10_efac','log10_equad'],
    'whiteband':    ['log10_efac','log10_equad','fL','fH'],
                    # ['log10_efac','A1','A2','A3','A4']
                    # ['log10_efac','log10_A1','log10_A2','log10_A3','log10_A4']
    'pwb':          ['Ared','gammared','log10_efac','log10_equad','log10_fH'],
    'exp':          ['log10_efac','log10_equad','log10_lambda','alpha'],
    'flat':         ['log10_efac','log10_equad','log10_lambda'],
    'pfec0':        ['log10_efac','log10_equad','Cquad','Ared','gammared']
}

def getmultinest(pulsar,model=None,dirname='chains',live=False,evidence=False):
    """Return list of parameters (according to pulsar and model name)
    and multinest equal-weights data."""

    if model is None:
        pulsar, model = pulsar.split('/')

    try:
        meta = json.load(open('{0}/{1}/{2}/{1}-{2}-summary.json'.format(dirname,pulsar,model),'r'))
        pars = meta['searchvars']
    except:
        print "Cannot find JSON record... switching to legacy parameter lists."

        pars = parlists[model]

        if 'nodm' in pulsar or 'EPTA_0.0' in pulsar:
            pars.append('log10_d1000')

    if '-' in model:
        shortmodel = model.split('-')[0]
    else:
        shortmodel = model

    if live:
        a = N.loadtxt('{0}/{1}/{2}/{1}-{3}-phys_live.points'.format(dirname,pulsar,model,shortmodel))

        if len(pars) + 1 != a.shape[1] - 1:
            raise 'Hm... multinest data does not have enough columns'
    else:
        a = N.loadtxt('{0}/{1}/{2}/{1}-{3}-post_equal_weights.dat'.format(dirname,pulsar,model,shortmodel))

        if len(pars) != a.shape[1] - 1:
            raise 'Hm... multinest data does not have enough columns'

    if evidence:
        lines = open('{0}/{1}/{2}/{1}-{3}-stats.dat'.format(dirname,pulsar,model,shortmodel)).readlines()

        return pars, a[:,:-1], float(re.search(r'Global Evidence:\s*(\S*)\s*\+/-\s*(\S*)',lines[0]).group(1))
        # for multinest 3.1
        # return pars, a[:,:-1], float(re.search(r'Global Log-Evidence           :\s*(\S*)\s*\+/-\s*(\S*)',lines[0]).group(1))
    else:
        return pars, a[:,:-1]

def plothist1(psr,flms,save=True):
    pars, pops, evs = {}, {}, {}

    for m in flms:
        pars[m], pops[m], evs[m] = getmultinest(pulsar=psr,model=m,evidence=True)

    lines = ['dotted','dashdot','dashed','solid']

    p = len(pars[flms[0]])
    P.figure(figsize=(16,3*(int((p-1)/4)+1)))
    for i in range(p):
        P.subplot(int((p-1)/4)+1,4,i+1)
        for m,l in zip(flms,lines):
        # for m,l in zip(flms[::-1],lines[::-1]):
            P.hist(pops[m][:,i],bins=50,normed=True,histtype='step',color='k',linestyle=l); P.hold(True)
        P.hold(False)
        P.xlabel(pars[m][i])

    if save:
        P.suptitle('{0}-{1}'.format(psr,flms[0]))
        P.savefig('figs/{0}-{1}.png'.format(psr,flms[0]))

def plothist2(psr,flm,save=True):
    pars, pops, evs = {}, {}, {}
    flms = [flm]

    for m in flms:
        pars[m], pops[m], evs[m] = getmultinest(pulsar=psr,model=m,evidence=True)

    data, thepars = pops[flms[0]].T, pars[flms[0]]

    m = len(thepars)

    fs = min((m-1)*4,16)
    P.figure(figsize=(fs,fs))

    for i in range(m-1):
        for j in range(i+1,m):
            P.subplot(m-1,m-1,(m-1)*i+j)
            # P.hist2d(data[j],data[i],bins=50,normed=True)

            bins = 50
            x,y = data[j],data[i]
            hrange = [[N.min(x),N.max(x)],[N.min(y),N.max(y)]]

            [h,xs,ys] = N.histogram2d(x,y,bins=bins,normed=True,range=hrange)
            P.contourf(0.5*(xs[1:]+xs[:-1]),0.5*(ys[1:]+ys[:-1]),h.T,cmap=P.get_cmap('YlOrBr')); P.hold(True)

            H,tmp1,tmp2 = N.histogram2d(x,y,bins=bins,range=hrange)

            import scipy.ndimage.filters as SNF
            H = SNF.gaussian_filter(H,sigma=1.5)

            H = H / len(x)                  # this is not correct with weights!
            Hflat = -N.sort(-H.flatten())   # sort highest to lowest
            cumprob = N.cumsum(Hflat)       # sum cumulative probability

            levels = [N.interp(level,cumprob,Hflat) for level in (0.6826,0.9547,0.9973)]

            xs = N.linspace(hrange[0][0],hrange[0][1],bins)
            ys = N.linspace(hrange[1][0],hrange[1][1],bins)

            P.contour(xs,ys,H.T,levels,colors='k',linestyles=('-','--','-.'),linewidths=2); P.hold(False)
            P.xlabel(thepars[j]); P.ylabel(thepars[i])

    if save:
        P.suptitle('{0}-{1}'.format(psr,flms[0]))
        P.savefig('figs/{0}-{1}-2.png'.format(psr,flms[0]))
