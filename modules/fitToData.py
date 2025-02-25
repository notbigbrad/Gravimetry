import numpy as np
import scipy
import matplotlib.pyplot as plt
from modules.model import sin as sin
from modules.model import physicalPendulum as model
from modules.errorProp import *

def fit(filename, l, lStd, trackingErr=0.05, phaseGuess=np.pi/2, cut=500, cameraRate=60, videoRate=60, focalLength=(24*1920)/8, doPlot=False, I0=False, r0=False, m=0.109):
    
    # Default values for non-physical pendulum mode
    if(I0==False): I0 = m*l**2
    if(r0==False): r0 = l
    
    # Import dataset
    time, x, _ = np.loadtxt(f'./data/{filename}.csv', delimiter=",", encoding="utf-8-sig").T

    # Remove initial outlier data
    time = time[cut::]*(cameraRate/videoRate) # convert from slo-mo seconds to real seconds
    x = x[cut::]
    
    # Apply imaging correction
    # focalLength (px) = focalLength (mm) * width (px) / width (mm)
    if (focalLength>0): x = np.arctan(x/focalLength)

    # Normalise
    x = x-np.min(x)
    x = x-np.max(x)/2
    x = x/np.max(x)
    
    # Initial guesses
    I = [1, 0.1, np.sqrt(9.81/l), phaseGuess] # A0, gamma, omega, phi

    # Bounds
    bounds = [[0.99,0.0001,0.1,-np.pi],[1.01,10,10,np.pi]] # bounds on the fitting function

    # Fit model
    optimal, covariance = scipy.optimize.curve_fit(model, time, x, p0=I, bounds=bounds, sigma=trackingErr, absolute_sigma=True, maxfev=1*10**9)

    # Check pendulum is lightly-damped
    if(optimal[1]**2 >= np.sqrt(optimal[2]**2 + optimal[1]**2)): quit("Scheiße: pendulum is not lightly-damped")
    
    # Find natural frequency
    # o^2 = o0^2 - gamma^2
    o0 = np.sqrt(optimal[2]**2 + optimal[1]**2)
    o0Std = sqrt(o0,optimal[2],optimal[1],np.sqrt(covariance[2,2]),np.sqrt(covariance[1,1]),np.sqrt(covariance[1,2]), "+")**2
   
    # Calculate g
    # o0^2 = mgro+ / I0
    # g = o0^2*I0 / mro+
    g = (o0**2*I0)/(m*r0)
    gStd = 0.01  # ERROR PROP CODE -------------------------------------
    print(f'g: {g} +- {gStd} ms^-2')

    # Calculate residuals
    r = x - model(np.linspace(np.min(time),np.max(time),len(time)), optimal[0], optimal[1], optimal[2], optimal[3])

    # Plot output
    tSpace = np.linspace(np.min(time),np.max(time),10000)

    if(doPlot):
        plt.figure(figsize=[10,7])
        plt.suptitle(filename)
        plt.axis("off")
        plt.subplot(211)
        plt.title("Data Plot with Fitted Model")
        plt.fill_between(time, x-trackingErr, x+trackingErr, color="lightgray")
        plt.scatter(time, x, color="red", marker="+", linewidths=1, label="Data")
        plt.plot(tSpace, model(tSpace, optimal[0], optimal[1], optimal[2], optimal[3]), label="Mathematical Pendulum Model")
        plt.plot(tSpace, sin(tSpace, np.sqrt(9.81/l), 0), "g--", label="Theoretical")
        plt.legend()
        plt.subplot(212)
        plt.title("Residual")
        plt.plot(time, x*0.05, color="lightgray")
        plt.plot(time, r, color="black")
        plt.show()

    return [g,gStd]