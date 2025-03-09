import numpy

def prop(m, c):
    M = numpy.array(m)
    C = numpy.array(c)
    return numpy.matmul(M.T,numpy.matmul(M,C))

def weightedMean(D, std):
    return numpy.mean(D), std/numpy.sqrt(len(D)) # requires constant std on all of population