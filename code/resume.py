#!/usr/bin/env python
# encoding: utf-8

import sys, os, glob, re

root = sys.argv[1]

last = glob.glob('../runs/chain-{0}.npy'.format(root))

if not last:
    print "Either you didn't run with this root name, or you've prepared for resume already."
    sys.exit(1)

prev = glob.glob('../runs/chain-{0}-*.npy'.format(root))

if prev:
    numbers = map(lambda file: int(re.search('../runs/chain-{0}-([0-9]*)\.npy'.format(root),file).group(1)),prev)
    next = max(numbers) + 1
else:
    next = 0

os.system("mv ../runs/chain-{0}.npy ../runs/chain-{0}-{1}.npy".format(root,next))
os.system("mv ../runs/lnprob-{0}.npy ../runs/lnprob-{0}-{1}.npy".format(root,next))
os.system("cp ../runs/resume-{0}.npy ../runs/resume-{0}-{1}.npy".format(root,next))
