#!/usr/bin/env python

# dat2 files made manually with
# $ for file in *.par; do tempo2 -gr plk -f $file `echo $file | sed 's/par/tim/g'` > `echo $file | sed 's/par/out/g'`; done
# $ for file in *.out; do tail -n 130 $file > `echo $file | sed 's/out/dat2/g'`; done

import sys, os, glob, re, random
import numpy as N
import ephem

workdir = sys.argv[1]

suffix = '.dat' # '.dat2'

pulsars = [re.sub('.dat','',filename) for filename in glob.glob(workdir + '/*.dat')]
loadone = N.loadtxt(pulsars[0] + suffix)

npulsars, ndata = len(pulsars), len(loadone)

meta = N.zeros((npulsars,),dtype=[('name','a32'),('ra','f8'),('dec','f8'),('designpars','i4'),('pars','i4')])
data = N.zeros((npulsars,ndata,3),'d')

# random.shuffle(pulsars)

design = []
designpars = 0

for i,pulsar in enumerate(pulsars):
    meta[i]['name'] = os.path.basename(pulsar)
    
    with open(pulsar + '.par','r') as pfile:
        for line in pfile:
            if 'RAJ' in line:
                raj  = line.split()[1]
            elif 'DECJ' in line:
                decj = line.split()[1]
                break
    
    pos = ephem.Equatorial(raj,decj,epoch='2000')
    meta[i]['ra']  = float(pos.ra)
    meta[i]['dec'] = float(pos.dec)
    meta[i]['designpars'] = designpars
    
    data[i,:,:] = N.loadtxt(pulsar + suffix)
    
    lines = open(pulsar + '.mat','r').readlines()
    times, pars = map(int,lines[0].split())
    
    design.append(N.array([map(float,line.split()) for line in lines[times+pars+4:]]))
    designpars = designpars + pars
    
    meta[i]['pars'] = pars
    

desi = N.zeros((npulsars*ndata,designpars),'d')

j = 0
for i,dp in enumerate(design):
    pars = dp.shape[1]
    desi[i*ndata:(i+1)*ndata,j:j+pars] = dp[:,:]
    j = j + pars

    # for three fitting pars, M would be
    #
    # |1 t_{11} t_{11}^2 0 0      0        ... |
    # |1 t_{12} t_{12}^2 0 0      0            |
    # |       ...             ...              |
    # |1 t_{1n} t_{1n}^2 0 0      0        ... |
    # |0 0      0        1 t_{21} t_{21}^2     |
    # |0 0      0        1 t_{22} t_{22}^2 ... |
    # |       ...               ...
    # |0 0      0        1 t_{2n} t_{2n}^2 ... |
    # |       ...               ...            |


N.save(workdir + '-meta.npy',meta)
N.save(workdir + '-data.npy',data)
N.save(workdir + '-desi.npy',desi)
