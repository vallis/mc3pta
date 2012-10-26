#!/usr/bin/env python

import sys, glob, os, re

if 'TEMPO2' not in os.environ:
    os.environ['TEMPO2']='/usr/local/share/tempo2'

workdir = sys.argv[1]

if workdir == 'epta':
    timfiles = ['0613_these.tim','1012_these.tim','1713-short.tim','1744all.tim','1909_these2.tim']
else:
    timfiles = glob.glob(workdir + '/*.tim')

for timfile in timfiles:
    parfile = re.sub('tim','par',timfile)
    datfile = re.sub('tim','dat',timfile)    
    matfile = re.sub('tim','mat',timfile)

    os.system('tempo2 -residuals -f {0} {1}'.format(parfile,timfile))
    os.rename('residuals.dat',datfile)

    os.system('tempo2 -gr designmatrix -f {0} {1}'.format(parfile,timfile))
    os.rename('designmatrix.txt',matfile)
