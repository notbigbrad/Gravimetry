import numpy as np
import scipy
import matplotlib.pyplot as plt
from modules.model import sin as sin
from modules.model import physicalPendulum as model

def fit(filename, l, lStd, trackingErr=0.05, phaseGuess=np.pi/2, cut=500, cameraRate=60, videoRate=60, focalLength=(24*1920)/8, doPlot=False):
    # Import dataset
    time, x, _ = np.loadtxt(f'./data/{filename}.txt', delimiter=",", encoding="utf-8-sig").T

    # Remove initial outlier data
    time = time[cut::]*(cameraRate/videoRate) # convert from slo-mo seconds to real seconds
    x = x[cut::]
    
    # Apply imaging correction
    # focalLength (px) = focalLength (mm) * width (px) / width (mm)
    x = np.arctan(x/focalLength)

    # Normalise the data
    x = x-np.min(x)
    x = x-np.max(x)/2
    x = x/np.max(x)
    
    # Initial guesses
    I = [1, 0.1, np.sqrt(9.81/l), phaseGuess] # A0, gamma, omega, phi

    # Bounds
    bounds = [[0.99,0.0001,0.1,-np.pi],[1.01,100,10,np.pi]] # bounds on the fitting function

    # Fit model
    optimal, covariance = scipy.optimize.curve_fit(model, time, x, p0=I, bounds=bounds, sigma=trackingErr, absolute_sigma=True, maxfev=1*10**9)

    # Calculate residuals
    space = np.linspace(np.min(time),np.max(time),len(time))
    r = x - model(space, optimal[0], optimal[1], optimal[2], optimal[3])

    # Plot output
    tSpace = np.linspace(np.min(time),np.max(time),10000)

    if(doPlot):
        plt.figure(figsize=[15,10])
        plt.title(filename)
        plt.axis("off")
        plt.subplot(211)
        plt.suptitle("Data Plot with Fitted Model")
        plt.fill_between(time, x-trackingErr, x+trackingErr, color="lightgray")
        plt.scatter(time, x, color="red", marker="+", linewidths=1, label="Data")
        plt.plot(tSpace, model(tSpace, optimal[0], optimal[1], optimal[2], optimal[3]), label="Mathematical Pendulum Model")
        plt.plot(tSpace, sin(tSpace, np.sqrt(9.81/l), 0), "g--", label="Theoretical")
        plt.legend()
        plt.subplot(212)
        plt.suptitle("Residual Plot")
        plt.plot(time, x*0.05, color="lightgray")
        plt.plot(time, r, color="black")
        plt.show()
        
    if(optimal[1]**2 >= optimal[2]**2 + optimal[1]**2): quit("Scheiße: pendulum is not lightly-damped")
    
    # Find natural frequency
    # o^2 = o0^2 - gamma^2
    o0 = np.sqrt(optimal[2]**2 + optimal[1]**2)
    o0Variance = covariance[2,2]*(optimal[2]/o0)**2 + covariance[1,1]*(optimal[1]/o0)**2 + 2*(optimal[1]*optimal[2]/o0**2)*covariance[1,2]
   
    # Calculate g
    # o^2 = g/l
    g = (o0**2)*l
    gVariance = g**2*((o0Variance/o0)**2+(lStd/l)**2)
    
    # Convert variance to std. dev.
    gStd = np.sqrt(gVariance)
    print(f'g: {g} +- {gStd} ms^-2')
    
    return [g,gStd]
