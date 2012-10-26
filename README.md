# mc3pta #

This is the JPL's group Bayesian-inference pipeline for the [first IPTA mock data challenge](http://www.ipta4gw.org/?page_id=89).

The pipeline estimates GWB and pulsar-noise parameters using Markov-Chain Monte Carlo integration with time-domain covariance-matrix evaluation of likelihoods, following van Haasteren et al. ([2009](http://adsabs.harvard.edu/abs/2009MNRAS.395.1005V)), automatically marginalizing over timing-solution uncertainties.

The integration is run with Goodman and Weare's ([2010](http://msp.org/camcos/2010/5-1/p04.xhtml)) affine-invariant ensemble sampler, as implemented in Foreman-Mackey et al.'s parallel `emcee` ([2012](http://adsabs.harvard.edu/abs/2012arXiv1202.3665F)). `tempo2` (version `cvs-2012-4`) was used to generate residuals and to dump the design matrix for the timing-solution fit, using the `DESIGNMATRIX` plugin. Covariance-matrix and likelihood routines are coded in Python, using the `numpy` and `scipy` vector and linear-algebra libraries, and the `multiprocessing` module for multi-core parallelization.

The `emcee` ensemble sampler evolves a population of walkers, and the last-step population can be used to approximate the posterior parameter distributions. Chain convergence can be assessed by plotting chains, integrated parameters, and by evaluating autocorrelation times.

## Requirements ##

* Python 2.7
* [numpy](http://numpy.scipy.org) and [scipy](http://numpy.scipy.org)
* [emcee](http://dan.iel.fm/emcee)
* [tempo2](http://www.atnf.csiro.au/research/pulsar/tempo2), including the `designmatrix` plugin
* [matplotlib](http://matplotlib.org), for plotting only

## Files ##

* `code/background.py`: the main Bayesian-inference script
* `code/constants.py`: a Python module that defines a number of simple physical constants
* `code/like.py`: a Python module that computes a likelihood based on the van Haasteren covariance-matrix approach
* `code/makearray.py`: a script that makes `numpy` data files from ASCII dumps of post-fit residuals and design matrices, as produced by `tempo2`
* `code/makeres.py`: a script that calls `tempo2` to dump residuals and design matrices
* `code/postprocess.py`: a script that computes simple statistics and makes plots from the `emcee` chains
* `code/resume.py`: a script that renames data files in `../runs` to allow a run to be restarted
* `code/util.py`: a Python module that defines two useful contexts (`timing`, to time a block of code, and `numpy_seterr`, to temporarily change the `numpy` error settings)

## Running the code ##

* You first need to translate the challenge `.par` and `.tim` files into a form suitable for the pipeline. This requires `tempo2`, and it's done by running the scripts `makeres.py` and `makearray.py` with a single argument, the directory that contains the `.par` and `.tim` files. The resulting `.npy` files (portable `numpy` arrays) should sit in directory `../tempo2` (with respect to the directory containing the scripts). For example, if the `.par` and `.tim` files for challenge open1 are in directory `../tempo2/open1`, you would do:

        $ python makeres.py ../tempo2/open1
        $ python makearray.py ../tempo2/open1

    These commands will create the files `open1-data.npy`, `open1-meta.npy`, and `open1-desi.npy` in `../tempo2`.
* You also need to create the directory `../runs`, which will contain the search results and plots.
* The script `background.py` performs Bayesian inference, and has the following help line:

        usage: background.py [-h] [-p P] [-s S] [-n N] [-w W] [-N N] [-l L] [-i] [-r] [-c C]
                             CHALLENGE

        positional arguments:
          CHALLENGE   challenge name

        optional arguments:
          -h, --help  show this help message and exit
          -p P        number of processors
          -s S        suffix for save file
          -n N        number of search parameters
          -w W        number of walkers
          -N N        number of iterations
          -r          resume run
          -c C        checkpoint interval

    For instance:

        $ python background.py open1 -s test -p 8 -n 74 -w 1000 -N 200

    which will use 8 cores to run a 74-parameter search (the GW amplitude and "alpha" exponent), plus red-noise parameters for all pulsars in the IPTA dataset. The search uses an `emcee` cloud of 1000 walkers, evolved through 200 steps. The resulting chain will be saved as a 200 x 1000 x 74 `numpy` array in `../runs/chain-open1-test-74.npy`; also `../runs/lnprob-open1-test-74.npy` contains the 200 x 1000 array of log posteriors, and `../runs/resume-open1-test-74.npy` the 1000 x 74 array describing the final cloud.
* Currently inference is possible with one parameter (the GW-background amplitude, with alpha exponent set to -2/3); two parameters (GW-background amplitude and exponent); four parameters (GW-background amplitude and exponent, plus red-noise amplitude and exponent common to all pulsars—note that these follow a different convention, as used in the IPTA MDC specification); 2 + 2 x #pulsars parameters (GW-background amplitude and exponent, plus individual red-noise amplitudes and exponents); 2 + 3 x #pulsars parameters (GW-background amplitude and exponent, plus individual red-noise amplitude, red-noise exponent, and "EFAC" multiplier for each pulsar).
* The search can be restarted by running `python resume.py open1-test-74`, and then giving the additional `-r` argument to the `background.py` command line. Running with the `-c` option will dump partial results (and a resume file) every `C` steps.
* The script `postprocess.py` outputs statistics and makes plots from the chain files:

        usage: postprocess.py [-h] [-n N] [-f F] [-p P] [-P] [-H] [-a] [-c] [-R] [-g]
                              CHALLENGE

        positional arguments:
          CHALLENGE   Challenge name

        optional arguments:
          -h, --help  show this help message and exit
          -n N        Number of search parameters
          -f F        Fraction of steps (or if negative, single step from end) to include
                      in statistics and histograms; defaults to the last step alone (-1)
          -p P        Which pulsar to plot red noise for [1-]
          -P          Save to PDF instead of showing to screen?
          -H          Plot histograms?
          -a          Plot evolution of averages?
          -c          Plot chains?
          -R          Make plot of all red noises (for 74/110-parameter runs)?

    In this case, we could run

        $ python postprocess.py open1-test -n 74 -H
