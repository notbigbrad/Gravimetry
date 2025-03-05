import numpy
import scipy

def prop(m, c):
    M = numpy.array(m)
    C = numpy.array(c)
    return numpy.matmul(M.T,numpy.matmul(M,C))