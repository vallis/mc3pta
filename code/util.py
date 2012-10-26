#!/usr/bin/env python
# encoding: utf-8
"""
util.py

Created by vallis on 2012-04-12.
Copyright (c) 2012 Caltech. All rights reserved.
"""

import time,contextlib
import numpy as N

@contextlib.contextmanager
def timing(name,debuglevel=1,debug=True):
    t0 = time.time()
    
    try:
        yield
    finally:
        if debug is True or debug >= debuglevel:
            print "%s: %.2f s" % (name,time.time() - t0)


@contextlib.contextmanager
def numpy_seterr(**kwargs):
    old_settings = N.seterr(**kwargs)
    
    try:
        yield
    finally:
        N.seterr(**old_settings)

