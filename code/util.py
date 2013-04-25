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


try:
    from IPython.core.display import HTML

    def htmltable(data,headings=None,format=None):
        tb =  '<html><table>'

        if headings:
            tb += '<tr>' + ''.join('<th>' + str(th) + '</th>' for th in headings) + '</tr>'

        for row in data:
            if format:
                tb += '<tr>' + ''.join(['<td style="text-align: right;">' + s + '</td>'
                                        for s in (format % tuple(row)).split(' ')]) + '</tr>'
            else:
                tb += '<tr>' + ''.join(['<td>' + str(d) + '</td>' for d in row]) + '</tr>'

        tb += '</table></html>'

        return HTML(tb)
except:
    pass
