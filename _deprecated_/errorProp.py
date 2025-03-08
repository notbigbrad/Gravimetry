import numpy as np

# ALL INPUTS TAKEN ARE STANDARD DEVIATIONS
# ALL OUTPUTS GIVEN ARE STANDARD DEVIATIONS

def sqrt(f,A,B,stdA,stdB,stdAB,sgn):
    if(sgn == "+"):
        return np.sqrt((A/f)**2 * stdA**2 + (B/f)**2 * stdB**2 + 2 * (A*B)/(f**2)*stdAB)
    if(sgn == "-"):
        return np.sqrt((A/f)**2 * stdA**2 + (B/f)**2 * stdB**2 - 2 * (A*B)/(f**2)*stdAB)
    
def div(f,A,B,stdA,stdB,stdAB):
    return f*np.sqrt((stdA/A)**2 + (stdB/B)**2 - 2*(stdAB)/(A*B))

def mul(f,A,B,stdA,stdB,stdAB):
    return f*np.sqrt((stdA/A)**2 + (stdB/B)**2 + 2*(stdAB)/(A*B))

def add(a,b,stdA,stdB,stdAB):
    return np.sqrt(a**2*stdA**2 + b**2*stdB**2 - 2*a*b*stdAB)

def squared(A, stdA):
    return (A**2*2*stdA)/A