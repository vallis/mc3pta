import sys, os, math, time, argparse, re
import numpy as N, numpy.lib.recfunctions as NLR
from mpi4py import MPI
import libstempo, libstempo.like

parser = argparse.ArgumentParser(description='Investigate timing solution with emcee and multinest.')

parser.add_argument('-n','--searchpars',dest='searchpars',default=None,
                    help='comma-separated list of search parameters [defaults to parfile fit pars]')
parser.add_argument('-d','--inputdir',  dest='inputdir',  default='.',
                    help='directory of par and tim files')
parser.add_argument('-f','--files',     dest='files',     default=None,
                    help='comma-separated par and tim files, or dataset name (e.g., nanograv_12, IPTA_13)')
parser.add_argument('-o','--outputdir', dest='outputdir', default='.',
                    help='output directory [defaults to .]')
parser.add_argument('-s','--suffix',    dest='suffix',    default=None,
                    help='save file suffix [defaults to none]')
parser.add_argument('-m','--comment',   dest='comment',   default='',
                    help='comment [defaults to none]')
parser.add_argument('-c','--config',    dest='config',    default=None,
                    help='configuration file [defaults to pulsar name + .ini]')
parser.add_argument('-w','--walkers',   dest='walkers',   default=1000, type=int,
                    help='number of walkers [defaults to 1000]')
parser.add_argument('-N','--steps',     dest='nsteps',    default=10000, type=int,
                    help='number of steps for MCMC [defaults to 10000]')
parser.add_argument('-v','--verbose',   dest='verbose',   action='store_true',
                    help='display verbose progress [defaults to false]')
parser.add_argument('-S','--sampler',   dest='sampler',   default='multinest',
                    help='sampler {multinest (default),emcee,PTMCMC}')
parser.add_argument(                    dest='pulsar',
                    help='pulsar name (no suffix)')

cfg = parser.parse_args()

# by default, we search over all parameters that are fit by tempo2
if not cfg.searchpars:
    cfg.searchpars = psr.fitpars
else:
    cfg.searchpars = libstempo.like.expandranges(cfg.searchpars.split(','))

# the output files will be of the form {outputdir}/{pulsar}[-{suffix}]-...
basename = '{0}/{1}{2}'.format(cfg.outputdir,cfg.pulsar,'-' + cfg.suffix if cfg.suffix else '')

parfile, timfile = libstempo.findpartim(pulsar=cfg.pulsar,dirname=cfg.inputdir,partimfiles=cfg.files)
pulsar = libstempo.tempopulsar(parfile=parfile,timfile=timfile)

for par in cfg.searchpars:
    if par in pulsar.fitpars:
        pulsar[par].fit = False

ll = libstempo.like.Loglike(pulsar,cfg.searchpars)
p0 = libstempo.like.Prior(pulsar,cfg.searchpars)

iamroot = MPI.COMM_WORLD.Get_rank() == 0

if cfg.config is None: cfg.config = cfg.pulsar + '.ini'
if os.path.isfile(cfg.config):
    if iamroot: print "Loading configuration file", cfg.config
    execfile(cfg.config)

if iamroot:
    p0.report()
    N.save(basename + '-meta.npy',NLR.merge_arrays((ll.meta,p0.meta),flatten=True))
    with open(basename + '-comment.txt','w') as cfile:
        cfile.write('Invocation: ' + ' '.join(sys.argv))
        if cfg.comment: cfile.write('\n' + cfg.comment)

improveml, resume = False, False

if cfg.sampler == 'emcee':
    import emcee, libstempo.emcee

    procs, skip = 8, 50

    if resume:
        init = N.load(basename + '-resume.npy')
    else:
        init = [N.random.random(len(cfg.searchpars)) for i in range(cfg.walkers)]

    # this function definition is needed because multiprocessing.Pool.map
    # requires a function that can be accessed by an import of the main module
    def logPL(xs):
        return libstempo.emcee.logPL(ll,p0,xs)

    sample = emcee.EnsembleSampler(len(init),len(cfg.searchpars),logPL,threads=procs)
    sample.run_mcmc(init,cfg.nsteps)

    libstempo.emcee.save(basename,sample,p0,skip)
elif cfg.sampler == 'multinest':
    import libstempo.multinest

    const_eff, efficiency = False, 0.8

    libstempo.multinest.run(LogLikelihood=ll,Prior=p0,n_dims=len(cfg.searchpars),
                            n_live_points=cfg.walkers,sampling_efficiency=efficiency,              # 0.3/0.8 for evidence/parameter evaluation
                            importance_nested_sampling=const_eff,const_efficiency_mode=const_eff,  # possible with newer MultiNest
                            outputfiles_basename=basename + '-',resume=resume,verbose=cfg.verbose,
                            init_MPI=False)                                                        # if init_MPI=False, I should be able to use MPI in Python
elif cfg.sampler == 'timing':
    import libstempo.emcee

    logPL = libstempo.emcee.logPL(ll,p0)
    def timing(iters):
        t0 = time.time()
        ret = logPL(N.random.random(len(cfg.searchpars)))
        return time.time() - t0

    t1 = timing(1)
    iters = int(max(10,10/t1))
    t1 = timing(iters)

    print "Timing: {0:.3f} s/iter (over {1} iters).".format(t1,iters)
    sys.exit(0)
else:
    raise NotImplementedError, "Unknown sampler {0}.".format(cfg.sampler)

if iamroot:
    if cfg.sampler == 'emcee':
        data = libstempo.emcee.load(basename)
    elif cfg.sampler == 'multinest':
        data = libstempo.multinest.load(basename)

    # move some or all of this business to like.py
    if improveml:
        import Simplex

        # same as emcee logp? But watch the offsets...
        def logp(xs):
            pprior = p0.premap(xs)
            prior = p0.prior(xs)
            return N.inf if not prior else -(math.log(prior) + ll.loglike(xs))

        optimizer = Simplex.Simplex(logp,
                                    [data[par].ml for par in cfg.searchpars],
                                    [0.1 * data[par].err for par in cfg.searchpars])

        minimum, error, iters = optimizer.minimize(maxiters=1000,monitor=0)

        ml = N.array(map(lambda x: (x,),minimum),dtype=[('ml','f16')])
        N.save(basename + '-meta.npy',NLR.merge_arrays((ll.meta,p0.meta,ml),flatten=True))

    ll.report(data)

    if cfg.sampler == 'multinest':
        libstempo.multinest.compress(basename)

# more things to report: logl and rmsres
