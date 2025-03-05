import numpy
import scipy
import scipy.stats

def prop(m, c):
    M = numpy.array(m)
    C = numpy.array(c)
    return numpy.matmul(M.T,numpy.matmul(M,C))

def wghtMn(D):


    return numpy.mean(D), numpy.std(D)