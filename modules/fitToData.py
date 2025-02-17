import numpy as np
import scipy
import matplotlib.pyplot as plt
import modules.mathematicalpendulum as mp

def fit(filename, l0, dl, trackingErr=0.05, phaseGuess=np.pi/2, cut=500, cameraRate=60, videoRate=60, doPlot=False):
    # Import dataset
    time, x, _ = np.loadtxt(f'./data/{filename}.txt', delimiter=",", encoding="utf-8-sig").T

    # Remove initial outlier data
    time = time[cut::]*(cameraRate/videoRate) # convert from slo-mo seconds to real seconds
    x = x[cut::]

    # Normalise the data
    x = x-np.min(x)
    x = x-np.max(x)/2
    x = x/np.max(x)

    # Initial guesses
    I = [1, 500, np.sqrt(9.81/l0), phaseGuess] # A0, gamma, omega, phi

    # Bounds
    bounds = [[0.99,1,0,-np.pi],[1.01,1000,10,np.pi]] # bounds on the fitting function

    optimal, covariance = scipy.optimize.curve_fit(mp.mathPendulum, time, x, p0=I, bounds=bounds, sigma=trackingErr, absolute_sigma=True, maxfev=1*10**9)

    # Plot output
    tSpace = np.linspace(np.min(time),np.max(time),10000)

    if(doPlot):
        plt.figure(figsize=[10,5])
        plt.fill_between(time, x-trackingErr, x+trackingErr, color="lightgray")
        plt.scatter(time, x, color="red", marker="+", linewidths=1, label="Data")
        plt.plot(tSpace, mp.mathPendulum(tSpace, optimal[0], optimal[1], optimal[2], optimal[3]), label="Mathematical Pendulum Model")
        plt.plot(tSpace, mp.sin(tSpace, np.sqrt(9.81/l0), 0), "g--", label="Theoretical")
        plt.legend()
        plt.show()

    # o^2 = g/l
    # => o^2 * l = g
    g = (optimal[2]**2)*l0
    gErr = (np.sqrt(np.diag(covariance))[2]**2)*l0
    print(f'g (w/o dl): {g} +- {gErr} ms^-2')
    # Take in length error
    absErr = g*np.sqrt((gErr/g)**2 + (dl/l0)**2)
    print(f'g: {g} +- {absErr} ms^-2')
    return [g,absErr]