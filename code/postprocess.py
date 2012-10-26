#!/usr/bin/env python
# encoding: utf-8
"""
postprocess.py

Created by vallis on 2012-06-14.
Copyright (c) 2012 California Institute of Technology
"""

from __future__ import division
import glob, sys, math
import numpy as N, matplotlib.pyplot as P


def getfiles(challenge,ndim):
    npyfiles = ( glob.glob('../runs/chain-{0}-{1}-*.npy'.format(challenge,ndim)) +
                 glob.glob('../runs/chain-{0}-{1}.npy'.format(challenge,ndim)) )

    if not npyfiles:
        print "Can't find file... what I see is:"
        for filename in glob.glob('../runs/chain-{0}-*'.format(challenge)):
            print filename
        sys.exit(1)

    a = N.load(npyfiles[0])
    for npyfile in npyfiles[1:]:
        a = N.concatenate((a,N.load(npyfile)),axis=1)

    return a


def load(challenge,ndim,last):
    a = getfiles(challenge,ndim)

    walker,steps,dims = a.shape

    if last != 1:
        if last < 0:
            last = steps + last

        if last < 1:
            print "Plotting %s walkers, %s/%s steps." % (walker,int(steps*float(last)),steps)
            steps = int(steps*float(last))
            a = a[:,-steps:,:]
        else:
            print "Plotting %s walkers, using step %s/%s." % (walker,last+1,steps)
            steps = 1
            a = a[:,int(last):int(last)+1,:]
    else:
        print "Plotting %s walkers, all %s steps." % (walker,steps)

    a[a[:,:,0] < 0,0] *= -1

    return a.reshape((walker*steps,dims))


def plot2plus(challenge,ndim=4,last=1,pulsar=1,save=False,hist=False,vara=False,varc=False,exp_gw='alpha'):
    # load data, keep only a 'last' fraction (or the -last step)
    a = load(challenge,ndim,last)

    if exp_gw == 'gamma':
        a[:,1] = 3 - 2*a[:,1]

    # display Bayesian statistics
    mean, err = N.mean(a,axis=0), N.sqrt(N.var(a,axis=0))

    if ndim in [2,4,74]:
        noise_skip = 2
    elif ndim == 110:
        noise_skip = 3

    for i in range(ndim):
        if i == 0:
            par = 'A_gw'
        elif i == 1:
            par = '{0}_gw'.format(exp_gw)
        elif (i - 2) % noise_skip == 0:
            par = 'A_red[%s]'     % int(i/noise_skip)
        elif (i - 2) % noise_skip == 1:
            par = 'alpha_red[%s]' % int(i/noise_skip)
        elif (i - 2) % noise_skip == 2:
            par = 'EFAC[%s]'      % int(i/noise_skip)

        print '{0:14}: {1:.3g} ± {2:.3g}    '.format(par,mean[i],err[i]),
        if i % noise_skip == 1:
            print

    if ndim in [74,110]:
        print '{0:14}: {1:.3g} ± {2:.3g}    '.format('A_red[mean]',    N.mean(mean[2::noise_skip]),N.sqrt(N.var(mean[2::noise_skip]))),
        print '{0:14}: {1:.3g} ± {2:.3g}    '.format('alpha_red[mean]',N.mean(mean[3::noise_skip]),N.sqrt(N.var(mean[3::noise_skip]))),

        if ndim == 110:
            print '{0:14}: {1:.3g} ± {2:.3g}    '.format('EFAC[mean]',     N.mean(mean[4::noise_skip]),N.sqrt(N.var(mean[4::noise_skip]))),

        print

    # plot histograms
    if hist:
        P.figure(2,figsize=(18,10)); P.clf()

        columns = 4 if ndim == 110 else 3

        P.subplot(2,columns,1)
        P.hist(a[:,0],bins=50,normed=True)
        P.title('{0}: marg. A'.format(challenge))
        P.xlabel('A');

        P.subplot(2,columns,2)
        P.hist(a[:,1],bins=50,normed=True)
        P.title('{0}: marg. {1}'.format(challenge,exp_gw))
        P.xlabel('alpha')

        P.subplot(2,columns,3)
        [h,x,y] = N.histogram2d(a[:,0],a[:,1],bins=50,normed=True)
        P.contourf(0.5*(x[1:]+x[:-1]),0.5*(y[1:]+y[:-1]),h.T)
        P.title('{0}: A vs. {1}'.format(challenge,exp_gw))
        P.xlabel('A'); P.ylabel('{0}'.format(exp_gw))
        P.colorbar()

        if ndim > 2:
            noise_par = 2 + noise_skip * (pulsar - 1)

            P.subplot(2,columns,4 if columns == 3 else 5)
            P.hist(a[:,noise_par],bins=50,normed=True)
            P.title('{0}: marg. Ared[{1}]'.format(challenge,pulsar))
            P.xlabel('Ared[{0}]'.format(pulsar));

            P.subplot(2,columns,5 if columns == 3 else 6)
            P.hist(a[:,noise_par+1],bins=50,normed=True)
            P.title('{0}: marg. alphared[{1}]'.format(challenge,pulsar))
            P.xlabel('alphared[{0}]'.format(pulsar))

            P.subplot(2,columns,6 if columns == 3 else 7)
            [h,x,y] = N.histogram2d(a[:,noise_par],a[:,noise_par+1],bins=50,normed=True)
            P.contourf(0.5*(x[1:]+x[:-1]),0.5*(y[1:]+y[:-1]),h.T)
            P.title('{0}: Ared vs. alphared[{1}]'.format(challenge,pulsar))
            P.xlabel('Ared[{0}]'.format(pulsar)); P.ylabel('alphared[{0}]'.format(pulsar))
            P.colorbar()

            if ndim == 110:
                P.subplot(2,columns,8)
                P.hist(a[:,noise_par+2],bins=50,normed=True)
                P.title('{0}: marg. EFAC[{1}]'.format(challenge,pulsar))
                P.xlabel('EFAC')

        P.suptitle('challenge {0}, {1}-parameter search'.format(challenge,ndim))

        if save:
            P.savefig('../runs/{0}-{1}-hist.pdf'.format(challenge,ndim))

    # further plots: analyze the convergence of conditional-mean estimators
    if vara:
        P.figure(3,figsize=(8,4)); P.clf()

        a = getfiles(challenge,ndim)

        a[a[:,:,0] < 0,0] *= -1     # correct negative A

        if exp_gw == 'gamma':
            a[:,:,1] = 3 - 2*a[:,:,1]

        amean, avar = N.mean(a[:,:,0],axis=0), N.var(a[:,:,0],axis=0)
        alphamean, alphavar = N.mean(a[:,:,1],axis=0), N.var(a[:,:,1],axis=0)

        # compute average acceptance ratios
        acc = 1 - N.sum(N.all(a[:,1:,:] == a[:,:-1,:],axis=2),axis=0)/a.shape[0]

        P.subplots_adjust(wspace=0.3)

        P.subplot(1,2,1)

        P.plot((amean + N.sqrt(avar))/1e-14,'r:'); P.hold(True)
        P.plot((amean - N.sqrt(avar))/1e-14,'r:')
        P.plot(amean/1e-14,'r')
        P.plot((alphamean + N.sqrt(alphavar)),'b:')
        P.plot((alphamean - N.sqrt(alphavar)),'b:')
        P.plot(alphamean,'b'); P.hold(False)

        P.xlabel('step index')
        P.ylabel('A/1e-14, {0}'.format(exp_gw))

        P.subplot(1,2,2)

        P.plot(acc,'g');

        P.xlabel('step index')
        P.ylabel('acceptance ratio')

        P.suptitle('challenge {0}, {1}-parameter search'.format(challenge,ndim))

        # meandump = N.zeros((len(amean),2),'d')
        # meandump[:,0] = amean[:]
        # meandump[:,1] = alphamean[:]
        # N.savetxt('{0}-{1}-means.txt'.format(challenge,ndim),meandump)

        if save:
            P.savefig('../runs/{0}-{1}-converge.pdf'.format(challenge,ndim))

    # further plots: chains
    if varc:
        P.figure(4,figsize=(12,10)); P.clf()
        a = getfiles(challenge,ndim)

        a[a[:,:,0] < 0,0] *= -1     # correct negative A

        if exp_gw == 'gamma':
            a[:,:,1] = 3 - 2*a[:,:,1]

        import acor

        for i in range(min(ndim,5 if ndim == 110 else 4)):
            if ndim == 110:
                P.subplot(2,3,i+1 if i < 2 else i+2)
            else:
                P.subplot(2,2,i+1)

            P.hold(True)

            walker,steps,dims = a.shape

            noise_par = 2 + noise_skip * (pulsar - 1)

            for j in range(walker):
                P.plot(a[j,:,i if i < 2 else noise_par + (i-2)],'b',alpha=1.0/math.sqrt(walker))

            if i == 0:
                par = 'A_gw'
            elif i == 1:
                par = '{0}_gw'.format(exp_gw)
            elif i == 2:
                par = 'A_red[%s]' % pulsar
            elif i == 3:
                par = 'alpha_red[%s]' % pulsar
            elif i == 4:
                par = 'EFAC[%s]' % pulsar

            t = acor.acor(a[:,:,i if i < 2 else noise_par + (i-2)])[0]
            t2 = acor.acor(a[:,int(a.shape[1]/2):,i if i < 2 else noise_par + (i-2)])[0]

            print "Autocorrelation for parameters {0}: {1:.1f}, {2:.1f} (at 50%)".format(par,t,t2)

            P.xlabel('step index (autocorrelation: {0:.2f})'.format(t))
            P.ylabel(par)

            P.hold(False)

        P.suptitle('challenge {0}, {1}-parameter search'.format(challenge,ndim))

        if save:
            P.savefig('../runs/{0}-{1}-chains.pdf'.format(challenge,ndim))


    if not save:
        P.show()


def plotrednoise(challenge,ndim=74,last=1,hist=False,vara=False,save=False):
    a = load(challenge,ndim,last)

    if ndim == 74:
        noise_skip = 2
    elif ndim == 110:
        noise_skip = 3

    P.figure(5,figsize=(12,12)); P.clf()
    P.subplots_adjust(wspace=0.3,hspace=0.4,bottom=0.05,top=0.95,left=0.05,right=0.95)

    for pulsar in range(36):
        P.subplot(6,6,pulsar+1)

        [h,x,y] = N.histogram2d(a[:,2 + noise_skip*pulsar],a[:,3 + noise_skip*pulsar],bins=50,normed=True)
        P.contourf(0.5*(x[1:]+x[:-1]),0.5*(y[1:]+y[:-1]),h.T)
        P.axis([1e-21,5e-21,1,2.5])
        P.xticks([1e-21,2e-21,3e-21,4e-21,5e-21])
        P.yticks([1,1.5,2,2.5])

    P.suptitle('challenge {0}, pulsar red noises'.format(challenge))

    if save:
        P.savefig('../runs/{0}-74-rednoise.pdf'.format(challenge))

        # P.title('{0}: Ared vs. alphared'.format(challenge))
        # P.xlabel('Ared'); P.ylabel('alphared')
        # P.colorbar()

    if hist:
        P.figure(6,figsize=(12,12)); P.clf()
        P.subplots_adjust(wspace=0.3,hspace=0.4,bottom=0.05,top=0.95,left=0.05,right=0.95)

        for pulsar in range(36):
            P.subplot(6,6,pulsar+1)
            P.hist(a[:,2 + noise_skip*pulsar],bins=50,normed=True)

        P.suptitle('Ared')

        P.figure(7,figsize=(12,12)); P.clf()
        P.subplots_adjust(wspace=0.3,hspace=0.4,bottom=0.05,top=0.95,left=0.05,right=0.95)

        for pulsar in range(36):
            P.subplot(6,6,pulsar+1)
            P.hist(a[:,3 + noise_skip*pulsar],bins=50,normed=True)

        P.suptitle('alphared')

    if vara:
        P.figure(8,figsize=(12,12)); P.clf()

        a = getfiles(challenge,ndim)

        for pulsar in range(36):
            P.subplot(6,6,pulsar+1)

            amean, avar = N.mean(a[:,:,2 + noise_skip*pulsar],axis=0), N.var(a[:,:,2 + noise_skip*pulsar],axis=0)
            alphamean, alphavar = N.mean(a[:,:,3 + noise_skip*pulsar],axis=0), N.var(a[:,:,3 + noise_skip*pulsar],axis=0)

            # P.plot((amean + N.sqrt(avar))/1e-21,'r:'); P.hold(True)
            # P.plot((amean - N.sqrt(avar))/1e-21,'r:')
            # P.plot(amean/1e-21,'r')
            P.plot((amean + N.sqrt(avar))/21,'r:'); P.hold(True)
            P.plot((amean - N.sqrt(avar))/21,'r:')
            P.plot(amean/21,'r')
            P.plot((alphamean + N.sqrt(alphavar)),'b:')
            P.plot((alphamean - N.sqrt(alphavar)),'b:')
            P.plot(alphamean,'b'); P.hold(False)

        P.suptitle('challenge {0}, {1}-parameter search'.format(challenge,ndim))

        if save:
            P.savefig('../runs/{0}-74-redconverge.pdf'.format(challenge))

    if not save:
        P.show()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot histograms for IPTA Monte Carlo runs')
    parser.add_argument('-n',type=int,default=74,  help='Number of search parameters')
    parser.add_argument('-f',type=float,default=-1,help='Fraction of steps (if neg, single step from end) to include; affects stats and hists, defaults to -1 (last step)')
    parser.add_argument('-p',type=int,default=1,   help='Which pulsar to plot red noise for [1-]')
    parser.add_argument('-P',action='store_true',  help='Save to PDF instead of showing to screen?')
    parser.add_argument('-H',action='store_true',  help='Plot histograms?')
    parser.add_argument('-a',action='store_true',  help='Plot evolution of averages?')
    parser.add_argument('-c',action='store_true',  help='Plot chains?')
    parser.add_argument('-R',action='store_true',  help='Make plot of all red noises (for 74/110-parameter runs)?')
    parser.add_argument('-g',action='store_true',  help='Use gamma_gw instead of alpha_gw')

    parser.add_argument('challenge',metavar='CHALLENGE',help='Challenge name')

    args = parser.parse_args()

    if args.R and args.n == 74 or args.n == 110:
        plotrednoise(args.challenge,args.n,last=args.f,save=args.P,hist=args.H,vara=args.a)
    else:
        plot2plus(args.challenge,args.n,last=args.f,pulsar=args.p,
                  save=args.P,hist=args.H,vara=args.a,varc=args.c,
                  exp_gw=('gamma' if args.g else 'alpha'))
