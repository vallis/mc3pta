# libstempo — a Python wrapper for tempo2 #

## Installation notes ##

To compile this wrapper you'll need to have the following installed:

* Python 2.6
* the gcc compiler suite
* a recent version of [tempo2](http://www.atnf.csiro.au/research/pulsar/tempo2/)
* [numpy](http://www.numpy.org/)
* version 0.19 of [Cython](http://www.cython.org/)

The makefile is currently written for OS X, but a simple replacement should make it suitable for Linux (see in the file itself). Also, the makefile assumes that tempo2 was installed to `/usr/local`.

Running `make` should make `libstempo.so`, which you can keep in your working directory, or copy to your Python `site-packages` directory.

## Documentation ##

For the moment, _in lieu_ of documentation, have a look at this [tutorial](http://nbviewer.ipython.org/urls/raw.github.com/vallis/mc3pta/master/stempo/libstempo-demo.ipynb).