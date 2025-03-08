import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy as sp
from modules.error_and_evalutation import evaluation_with_error
from modules.position import *
from modules.model import *
from modules.error import *
from modules.errorProp import *
import concurrent.futures
from joblib import Parallel, delayed
from datetime import datetime as dt
plt.show = lambda : 0 # dissable plot output

def optimize_curve(i, arrays, time, data, p0, bounds):
    optimal, covariance = scipy.optimize.curve_fit(lambda t, thet0, om0, g, b: physicalODE(t, thet0, om0, g, b, m=arrays[i][0], r=arrays[i][1], I=arrays[i][2]), time, data, p0=p0, bounds=bounds)
    print(optimal[2], end='\r', flush=True)
    return optimal[2], np.sqrt(covariance[2, 2])

def prop(func, time, data, bounds, p0, constants, constantStd):
    
    print(f'Started {dt.now()}')
    n = int(1e6)
    
    arrays = []
    
    for i in range(len(constants)):
        arrays.append(np.random.normal(constants[i], constantStd[i], n))
    
    arrays = np.array(arrays).T
    
    results = []
    errors = []
    
    # for i in range(len(arrays)):
    #     optimal, covariance = scipy.optimize.curve_fit(lambda t, thet0, om0, g, b : func(t, thet0, om0, g, b, m=arrays[i][0], r=arrays[i][1], I=arrays[i][2]), time, data, p0=p0, bounds=bounds, maxfev=1*10**9)
    #     results.append(optimal[2])
    #     errors.append(np.sqrt(covariance[2,2]))
    
    results, errors = zip(*Parallel(n_jobs=22)(
    delayed(optimize_curve)(i, arrays, time, data, p0, bounds) for i in range(len(arrays))
    ))
    # with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    #     futures = [executor.submit(optimize_curve, i, arrays, time, data, p0, bounds) for i in range(len(arrays))]
        
    #     for future in concurrent.futures.as_completed(futures):
    #         optimal_2, error_2 = future.result()
    #         results.append(optimal_2)
    #         errors.append(error_2)
        
    # Collate all data
    tSpace = np.linspace(np.min(results),np.max(results),len(results))
        
    optimal, covariance = scipy.optimize.curve_fit(f, tSpace, results, sigma=errors, absolute_sigma=True, maxfev=1*10**9)    
    
    print(f'Done {dt.now()}')
    
    # Plot output
    plt.figure(figsize=[14,7])
    plt.title("Monte Carlo")
    plt.errorbar(tSpace, results, yerr=errors, color="red")
    plt.plot(tSpace, f(tSpace, optimal[0]))
    plt.show()
    
    return optimal, covariance