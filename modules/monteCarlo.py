import numpy as np
import scipy
import os
from modules.model import *
from joblib import Parallel, delayed
from datetime import datetime as dt

def prop(func, time, data, bounds, p0, constants, constantStd, n=1e5, processors=os.process_cpu_count()-1, **kwargs):

    def ODE_wrapper(i, arrays, x, y):
        optimal, covariance = scipy.optimize.curve_fit(lambda t, *fittedParams: func(t, *fittedParams, m=arrays[i][0], r=arrays[i][1], I=arrays[i][2]), x, y, p0=p0, bounds=bounds)
        print(optimal[2], end='\r', flush=True)
        return optimal[2], np.sqrt(covariance[2, 2])
        
    print(f'Started {dt.now()}')
    
    n = int(n)
    
    noise = []
    
    for i in range(len(constants)):
        noise.append(np.random.normal(constants[i], constantStd[i], n))
    
    noise = np.array(noise).T
    
    results = []
    errors = []

    results, errors = zip(*Parallel(n_jobs=processors)(
        delayed(ODE_wrapper)(i, noise, time, data) for i in range(len(noise))
    ))
    
    print(f'Done {dt.now()}')
    
    return results, errors, [np.mean(results), scipy.stats.sem(results)]