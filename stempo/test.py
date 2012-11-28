import libstempo as T
import numpy as N
import scipy.linalg as SL

# pulsar = T.tempopulsar("../tempo2/open1/J0030+0451")

pulsar = T.tempopulsar("../nanograv/par/B1855+09_NANOGrav_dfg+12",
                       "../nanograv/tim/B1855+09_NANOGrav_dfg+12",
                       debug=True)

pars = pulsar.parameters

print "Parameters (%s):" % len(pars)
print pulsar.numpypars()

des = pulsar.designmatrix()
print "Design matrix shape (includes constant phase):", des.shape
print des

data = pulsar.data()
print "Data shape:", data.shape
print data
