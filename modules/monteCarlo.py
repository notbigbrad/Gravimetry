import numpy as np
import scipy
from modules.model import *
from joblib import Parallel, delayed
from datetime import datetime as dt

def optimize_curve(i, arrays, time, data, p0, bounds):
    optimal, covariance = scipy.optimize.curve_fit(lambda t, thet0, om0, g, b: physicalODE(t, thet0, om0, g, b, m=arrays[i][0], r=arrays[i][1], I=arrays[i][2]), time, data, p0=p0, bounds=bounds)
    print(optimal[2], end='\r', flush=True)
    return optimal[2], np.sqrt(covariance[2, 2])

def prop(func, time, data, bounds, p0, constants, constantStd):
    
    print(f'Started {dt.now()}')
    n = int(1e4)
    
    arrays = []
    
    for i in range(len(constants)):
        arrays.append(np.random.normal(constants[i], constantStd[i], n))
    
    arrays = np.array(arrays).T
    
    results = []
    errors = []

    results, errors = zip(*Parallel(n_jobs=24)(
    delayed(optimize_curve)(i, arrays, time, data, p0, bounds) for i in range(len(arrays))
    ))
        
    # Collate all data
    final = [np.mean(results), scipy.stats.sem(results)]
        
    # optimal, covariance = scipy.optimize.curve_fit(f, tSpace, results, sigma=errors, absolute_sigma=True, maxfev=1*10**9)
    
    print(f'Done {dt.now()}')
    
    return results, errors, final