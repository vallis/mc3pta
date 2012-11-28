import os, math

from libc cimport stdlib, stdio

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
        long double bat        # barycentric arrival time
        long double prefitResidual
        long double residual
        double toaErr          # error on TOA (in us)
        double toaDMErr        # error on TOA due to DM (in us)

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

cdef class tempopulsar:
    cdef int npsr
    cdef pulsar *psr
    cdef parameter *params

    cpdef public object parameters, allparameters, ndim, nobs, jumpval, jumperr, debug

    def __cinit__(self,parfile,timfile=None,warnings=False,debug=False):
        """Create a tempo2 pulsar object from .par and .tim files;
        if only the former is given, will replace the suffix as appropriate."""

        global MAX_PSR, MAX_OBSN

        cdef char parFile[MAX_PSR_VAL][MAX_FILELEN]
        cdef char timFile[MAX_PSR_VAL][MAX_FILELEN]

        self.npsr = 1
        self.psr = <pulsar *>stdlib.malloc(sizeof(pulsar)*MAX_PSR_VAL)

        MAX_PSR, MAX_OBSN = 1, 5000     # to save memory, only allocate space for this many pulsars and observations
        initialise(self.psr,1)          # 1 for no warnings

        if timfile is None:
            timfile = parfile

        if parfile[-4:] != '.par':      # accept parfile with or without suffix
            parfile = parfile + '.par'

        if timfile[-4:] != '.tim':
            timfile = timfile + '.tim'

        if not os.path.isfile(parfile) or not os.path.isfile(timfile):
            raise IOError, "Cannot find parfile (%s) or timfile (%s)!" % (parfile,timfile)

        stdio.sprintf(parFile[0],"%s",<char *>parfile);
        stdio.sprintf(timFile[0],"%s",<char *>timfile);

        readParfile(self.psr,parFile,timFile,self.npsr);   # load the parameters    (all pulsars)
        readTimfile(self.psr,timFile,self.npsr);           # load the arrival times (all pulsars)

        self.psr.rescaleErrChisq = 0    # do not rescale fit errors by sqrt(red. chisq)

        if not warnings:
            self.psr.noWarnings = 2         # do not show some warnings

        preProcess(self.psr,self.npsr,0,NULL)
        formBatsAll(self.psr,self.npsr)

        self.debug = debug

        self.readpars()

        updateBatsAll(self.psr,self.npsr)
        formResiduals(self.psr,self.npsr,1)     # 1 to remove the mean

        doFit(self.psr,self.npsr,0)

        updateBatsAll(self.psr,self.npsr)
        formResiduals(self.psr,self.npsr,1)     # 1 to remove the mean

        self.savejumps()

        self.ndim = len(self.parameters)
        self.nobs = self.psr[0].nobs

    def readpars(self):
        """Used internally to read the names and values of parameters from the tempo2 pulsar structure."""
        
        self.params = self.psr[0].param

        self.parameters = []            # fit parameters, including jumps
        self.allparameters = []         # all parameters

        for ct in range(MAX_PARAMS):
            for subct in range(self.params[ct].aSize):
                if self.params[ct].fitFlag[subct] == 1:
                    self.parameters.append((self.params[ct].shortlabel[subct],ct,subct))

                self.allparameters.append((self.params[ct].shortlabel[subct],ct,subct))

        for ct in range(1,self.psr[0].nJumps+1):  # jump 1 not used...
            if self.psr[0].fitJump[ct] == 1:
                self.parameters.append(('JUMP{0}'.format(ct),-1,ct))

            self.allparameters.append(('JUMP{0}'.format(ct),-1,ct))

        # the designmatrix plugin also adds extra parameters for sinusoidal whitening
        # but they don't seem to be used in the EPTA analysis
        # if(pPsr->param[param_wave_om].fitFlag[0]==1)
        #     nPol += pPsr->nWhite*2-1;

    def data(self):
        """Returns a numpy array (quad precision) of the timing data:
            col 1: timestamps in days
            col 2: post-fit residuals in seconds
            col 3: TOA errors in seconds"""
        
        ret = numpy.zeros((self.nobs,3),dtype='f16')

        cdef long double epoch = self.psr[0].param[param_pepoch].val[0]
        obsns = self.psr[0].obsn

        cdef int i
        for i in range(self.nobs):
            ret[i,0] = numpy.longdouble(obsns[i].bat - epoch)       # days (multiply by 86400.0 for seconds)
            ret[i,1] = numpy.longdouble(obsns[i].residual)          # seconds
            ret[i,2] = 1.0e-6 * numpy.longdouble(obsns[i].toaErr)   # seconds

        return ret

    def designmatrix(self):
        """Returns the tempo2 design matrix as a numpy 2D array."""

        # print "Check pepoch enum:", param_pepoch

        cdef numpy.ndarray[double,ndim=2] ret = numpy.zeros((self.nobs,self.ndim+1),'d')
        cdef long double epoch = self.psr[0].param[param_pepoch].val[0]
        obsns = self.psr[0].obsn

        cdef int i
        for i in range(self.nobs):
            # always fit for arbitrary offset (the "+1")
            FITfuncs(obsns[i].bat - epoch,&ret[i,0],self.ndim+1,&self.psr[0],i)
        return ret

    def rebase(self):
        """Reset prefit-parameter values and errors (used as origin and scale of search) to post-fit values.
        Note: does not affect jumps, which are always used post-fit."""

        for par in self.parameters:
            ct,subct = par[1:]

            if ct != -1:
                self.params[ct].prefit[subct]    = self.params[ct].val[subct]
                self.params[ct].prefitErr[subct] = self.params[ct].err[subct]

    def savejumps(self):
        """Used internally to save the values and errors of phase jumps."""
        
        self.jumpval = numpy.zeros(self.psr[0].nJumps,'f16')
        self.jumperr = numpy.zeros(self.psr[0].nJumps,'f16')

        for par in self.parameters:
            ct,subct = par[1:]

            if ct == -1:
                self.jumpval[subct-1] = numpy.longdouble(self.psr[0].jumpVal[subct]   )
                self.jumperr[subct-1] = numpy.longdouble(self.psr[0].jumpValErr[subct])

    def numpypars(self,all=False):
        """Returns a python record array containing the name as well as the pre-fit
        and post-fit values and errors of the fit parameters. If all is set to True,
        of all the .tim-file parameters."""
        
        pars = self.allparameters if all else self.parameters

        ndim = len(pars)
        maxlen = max(len(par[0]) for par in pars)

        ret = numpy.zeros((ndim,),dtype=[('name','a' + str(maxlen)),('val','f16'),('err','f16'),('pval','f16'),('perr','f16')])

        for i,par in enumerate(pars):
            ret[i]['name'] = par[0]

            if par[1] == -1:    # it's a jump!
                ret[i]['val'] = ret[i]['pval'] = numpy.longdouble(self.psr.jumpVal[par[2]])
                ret[i]['err'] = ret[i]['perr'] = numpy.longdouble(self.psr.jumpValErr[par[2]])
            else:
                ret[i]['val']  = numpy.longdouble(self.params[par[1]].val[par[2]]      )
                ret[i]['err']  = numpy.longdouble(self.params[par[1]].err[par[2]]      )

                ret[i]['pval'] = numpy.longdouble(self.params[par[1]].prefit[par[2]]   )
                ret[i]['perr'] = numpy.longdouble(self.params[par[1]].prefitErr[par[2]])

        return ret

    # def fitpars(self,all=False):
    #     return dict((label,(self.params[ct].val[subct],self.params[ct].err[subct]))
    #                 for (label,ct,subct) in (self.allparameters if all else self.parameters))
    #
    # def prefitpars(self,all=False):
    #     return dict((label,(self.params[ct].prefit[subct],self.params[ct].prefitErr[subct]))
    #                 for (label,ct,subct) in (self.allparameters if all else self.parameters))

    def fit(self):
        """Used internally to run a tempo2 fit."""

        doFit(self.psr,self.npsr,0)
        updateBatsAll(self.psr,self.npsr)
        formResiduals(self.psr,self.npsr,1)     # 1 to remove the mean

    def __dealloc__(self):
        """Called by cython to destroy pulsar storage structures."""
        
        for i in range(self.npsr):
            destroyOne(&(self.psr[i]))

