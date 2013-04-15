import os, math, re, collections

from libc cimport stdlib, stdio
from cython cimport view

import numpy
cimport numpy

cdef extern from "tempo2.h":
    enum: MAX_PSR_VAL
    enum: MAX_FILELEN
    enum: MAX_PARAMS
    enum: MAX_JUMPS
    enum: param_pepoch

    int MAX_PSR, MAX_OBSN

    ctypedef struct parameter:
        char **label
        char **shortlabel
        long double *val
        long double *err
        int  *fitFlag
        int  *paramSet
        long double *prefit
        long double *prefitErr
        int aSize

    ctypedef struct observation:
        long double sat        # site arrival time
        long double bat        # barycentric arrival time
        long double prefitResidual
        long double residual
        double toaErr          # error on TOA (in us)
        double toaDMErr        # error on TOA due to DM (in us)
        double freq            # frequency of observation (in MHz)
        double freqSSB         # frequency of observation in barycentric frame (in Hz)

    ctypedef struct pulsar:
        parameter param[MAX_PARAMS]
        observation *obsn
        int nobs
        int rescaleErrChisq
        int noWarnings
        double fitChisq
        int nJumps
        double jumpVal[MAX_JUMPS]
        int fitJump[MAX_JUMPS]
        double jumpValErr[MAX_JUMPS]

    void initialise(pulsar *psr, int noWarnings)
    void destroyOne(pulsar *psr)

    void readParfile(pulsar *psr,char parFile[][MAX_FILELEN],char timFile[][MAX_FILELEN],int npsr)
    void readTimfile(pulsar *psr,char timFile[][MAX_FILELEN],int npsr)

    void preProcess(pulsar *psr,int npsr,int argc,char *argv[])
    void formBatsAll(pulsar *psr,int npsr)
    void updateBatsAll(pulsar *psr,int npsr)                    # what's the difference?
    void formResiduals(pulsar *psr,int npsr,int removeMean)
    void doFit(pulsar *psr,int npsr,int writeModel)

    void FITfuncs(double x,double afunc[],int ma,pulsar *psr,int ipos)

    int turn_hms(double turn,char *hms)


# TO DO: should I do also pre-fit values?
cdef class tempopar:
    cdef public object name

    cdef int _isjump
    cdef void *_val
    cdef void *_err
    cdef int *_fitFlag

    def __init__(self,*args,**kwargs):
        raise TypeError("This class cannot be instantiated from Python.")

    property val:
        def __get__(self):
            if not self._isjump:
                return numpy.longdouble((<long double*>self._val)[0])
            else:
                return float((<double*>self._val)[0])
        def __set__(self,value):
            if not self._isjump:
                (<long double*>self._val)[0] = value    # can we set it to numpy.longdouble?
            else:
                (<double*>self._val)[0] = value

    property err:
        def __get__(self):
            if not self._isjump:
                return numpy.longdouble((<long double*>self._err)[0])
            else:
                return float((<double*>self._err)[0])
        def __set__(self,value):
            if not self._isjump:
                (<long double*>self._err)[0] = value
            else:
                (<double*>self._err)[0] = value

    property fit:
        def __get__(self):
            return True if self._fitFlag[0] else False
        def __set__(self,value):
            self._fitFlag[0] = 1 if value else 0

    def __str__(self):
        # TO DO: proper precision handling
        return '%s: %g +/- %g' % (self.name,self.val,self.err)

# since the __init__ for extension classes must have a Python signature,
# we use a factory function to initialize its attributes to pure-C objects

cdef create_tempopar(parameter par,int subct):
    cdef tempopar newpar = tempopar.__new__(tempopar)

    newpar.name = str(par.shortlabel[subct])

    newpar._isjump = 0

    newpar._val = &par.val[subct]
    newpar._err = &par.err[subct]
    newpar._fitFlag = &par.fitFlag[subct]

    return newpar

cdef create_tempojump(pulsar *psr,int ct):
    cdef tempopar newpar = tempopar.__new__(tempopar)

    newpar.name = 'JUMP{0}'.format(ct)

    newpar._isjump = 1

    newpar._val = &psr.jumpVal[ct]
    newpar._err = &psr.jumpValErr[ct]
    newpar._fitFlag = &psr.fitJump[ct]

    return newpar

class prefitpar(object):
    def __init__(self,name,val,err):
        self.__dict__['name'] = name
        self.__dict__['val']  = val
        self.__dict__['err']  = err

    def __setattr__(self,par,val):
        raise TypeError, "Cannot write to prefit parameters."

    def __str__(self):
        # TO DO: proper precision handling
        return '%s: %g +/- %g' % (self.name,self.val,self.err)

# this is a Cython extension class; the benefit is that it can hold C attributes,
# but all attributes must be defined in the code

cdef class tempopulsar:
    cdef int npsr           # number of pulsars
    cdef pulsar *psr        # array of pulsar structures
    cpdef object pardict    # dictionary of parameter proxies
    cpdef public object prefit     # dictionary of pre-fit parameters
    cpdef public int nobs   # number of observations (public)

    # TO DO: is cpdef required here?
    cpdef jumpval, jumperr

    def __cinit__(self,parfile,timfile=None,warnings=False,fixangularerror=True):
        # initialize

        global MAX_PSR, MAX_OBSN

        self.npsr = 1
        self.psr = <pulsar *>stdlib.malloc(sizeof(pulsar)*MAX_PSR_VAL)

        MAX_PSR, MAX_OBSN = 1, 5000     # to save memory, only allocate space for this many pulsars and observations
        initialise(self.psr,1)          # 1 for no warnings

        # read par and tim file

        self._readfiles(parfile,timfile)

        # set tempo2 flags

        self.psr.rescaleErrChisq = 0        # do not rescale fit errors by sqrt(red. chisq)

        if not warnings:
            self.psr.noWarnings = 2         # do not show some warnings

        # preprocess the data

        preProcess(self.psr,self.npsr,0,NULL)
        formBatsAll(self.psr,self.npsr)

        # create parameter proxies, copy prefit values

        self.nobs = self.psr[0].nobs
        self._readpars(fixangularerror=fixangularerror)

        # always do a fit...

        updateBatsAll(self.psr,self.npsr)
        formResiduals(self.psr,self.npsr,1)     # 1 to remove the mean

        doFit(self.psr,self.npsr,0)

    def __dealloc__(self):
        for i in range(self.npsr):
            destroyOne(&(self.psr[i]))

    def _readfiles(self,parfile,timfile):
        cdef char parFile[MAX_PSR_VAL][MAX_FILELEN]
        cdef char timFile[MAX_PSR_VAL][MAX_FILELEN]

        if timfile is None:
            timfile = re.sub('\.par$','.tim',parfile)

        if not os.path.isfile(parfile) or not os.path.isfile(timfile):
            raise IOError, "Cannot find parfile (%s) or timfile (%s)!" % (parfile,timfile)

        stdio.sprintf(parFile[0],"%s",<char *>parfile);
        stdio.sprintf(timFile[0],"%s",<char *>timfile);

        readParfile(self.psr,parFile,timFile,self.npsr);   # load the parameters    (all pulsars)
        readTimfile(self.psr,timFile,self.npsr);           # load the arrival times (all pulsars)

    def _readpars(self,fixangularerror=True):
        cdef parameter *params = self.psr[0].param

        # create live proxies for all the parameters
        # and collect the prefit values of the parameters

        self.pardict = collections.OrderedDict()
        self.prefit = collections.OrderedDict()

        for ct in range(MAX_PARAMS):
            for subct in range(params[ct].aSize):
                newpar = create_tempopar(params[ct],subct)
                self.pardict[newpar.name] = newpar
                self.prefit[newpar.name] = prefitpar(newpar.name,
                                                     numpy.longdouble(params[ct].prefit[subct]),
                                                     numpy.longdouble(params[ct].prefitErr[subct]))

        for ct in range(1,self.psr[0].nJumps+1):  # jump 1 in the array not used...
            newpar = create_tempojump(&self.psr[0],ct)
            self.pardict[newpar.name] = newpar
            self.prefit[newpar.name] = prefitpar(newpar.name,
                                                 numpy.longdouble(self.psr[0].jumpVal[ct]),
                                                 numpy.longdouble(self.psr[0].jumpValErr[ct]))

        # TODO: it should also not be possible to replace or alter prefit,
        #       or to replace prefit.vals and prefit.errs

        self.prefit.vals = numpy.fromiter((self.prefit[par].val for par in self.pars),'f16')
        self.prefit.vals.flags.writeable = False

        self.prefit.errs = numpy.fromiter((self.prefit[par].err for par in self.pars),'f16')
        self.prefit.errs.flags.writeable = False

        # the designmatrix plugin also adds extra parameters for sinusoidal whitening
        # but they don't seem to be used in the EPTA analysis
        # if(pPsr->param[param_wave_om].fitFlag[0]==1)
        #     nPol += pPsr->nWhite*2-1;

    def __getitem__(self,key):
        return self.pardict[key]

    property pars:
        def __get__(self):
            return tuple(key for key in self.pardict if self.pardict[key].fit)

    property allpars:
        def __get__(self):
            return tuple(self.pardict)

    property vals:
        def __get__(self):
            ret = numpy.fromiter((self.pardict[par].val for par in self.pars),'f16')
            ret.flags.writeable = False
            return ret

        def __set__(self,values):
            for par,value in zip(self.pars,values):
                self.pardict[par].val = value
                self.pardict[par].err = 0

    property errs:
        def __get__(self):
            ret = numpy.fromiter((self.pardict[par].err for par in self.pars),'f16')
            ret.flags.writeable = False
            return ret

    # the best way to access prefit pars would be through the same interface:
    # psr.prefit['parname'].val, psr.prefit['parname'].err, perhaps even psr.prefit.cols
    # since the prefit values don't change, it's OK for psr.prefit to be a static attribute

    # number of active fit parameters
    property ndim:
        def __get__(self):
            return sum(self.pardict[par].fit for par in self.pardict)

    # TOAs in days (numpy.longdouble array)
    property toas:
        def __get__(self):
            cdef long double [:] _toas = <long double [:self.nobs]>&(self.psr[0].obsn[0].bat)
            _toas.strides[0] = sizeof(observation)

            return numpy.asarray(_toas)

    # site arrival times; divide residuals by 86400.0 to subtract
    property stoas:
        def __get__(self):
            cdef long double [:] _stoas = <long double [:self.nobs]>&(self.psr[0].obsn[0].sat)
            _stoas.strides[0] = sizeof(observation)

            return numpy.asarray(_stoas)

    # return TOA errors in microseconds (numpy.double array)
    property toaerrs:
        def __get__(self):
            cdef double [:] _toaerrs = <double [:self.nobs]>&(self.psr[0].obsn[0].toaErr)
            _toaerrs.strides[0] = sizeof(observation)

            return numpy.asarray(_toaerrs)

    # frequencies in MHz (numpy.double array)
    property freqs:
        def __get__(self):
            cdef double [:] _freqs = <double [:self.nobs]>&(self.psr[0].obsn[0].freq)
            _freqs.strides[0] = sizeof(observation)

            return numpy.asarray(_freqs)

    # residuals in seconds
    def residuals(self):
        cdef int i
        cdef numpy.ndarray[longdouble,ndim=1] ret = numpy.zeros(self.nobs,'f16')

        cdef observation *obsns = self.psr[0].obsn

        updateBatsAll(self.psr,self.npsr)
        formResiduals(self.psr,self.npsr,1)     # 1 to remove the mean

        for i in range(self.nobs):
            ret[i] = obsns[i].residual

        return ret

    # tempo2 design matrix as numpy array [nobs x ndim]
    def designmatrix(self):
        cdef int i
        cdef numpy.ndarray[double,ndim=2] ret = numpy.zeros((self.nobs,self.ndim+1),'d')

        cdef long double epoch = self.psr[0].param[param_pepoch].val[0]
        cdef observation *obsns = self.psr[0].obsn

        updateBatsAll(self.psr,self.npsr)

        for i in range(self.nobs):
            # always fit for arbitrary offset... that's the "+1"
            FITfuncs(obsns[i].bat - epoch,&ret[i,0],self.ndim+1,&self.psr[0],i)

        return ret

    # run tempo2 fit
    def fit(self):
        updateBatsAll(self.psr,self.npsr)
        formResiduals(self.psr,self.npsr,1)     # 1 to remove the mean

        doFit(self.psr,self.npsr,0)

    # utility function
    def rd_hms(self):
        cdef char retstr[256]

        ret = {}
        for i,par in enumerate(self.parameters):
            if par[0] in ['RAJ','DECJ']:
                turn_hms(float(self.params[par[1]].val[par[2]]/(2*math.pi)),<char *>retstr)
                ret[par[0]] = retstr

        return ret['RAJ'], ret['DECJ']

    # legacy support; abs only
    def logL(self,dxs=None,abs=False):
        if dxs is not None:
            if not abs:
                raise NotImplementedError, 'pulsar.logL() works only with abs=True'

            self.vals = dxs

        res, err = self.residuals(), self.toaerrs

        return -0.5 * numpy.sum(res * res / (1e-12 * err * err))
