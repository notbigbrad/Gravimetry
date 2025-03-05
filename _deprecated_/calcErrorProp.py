import numpy

def prop(m, c):
    M = numpy.array(m)
    C = numpy.array(c)
    # M C M^T
    # M is nx1, C is nxn, M^T is 1xn
    return numpy.matmul(M.T,numpy.matmul(M,C))