import numpy as np
import scipy
import matplotlib.pyplot as plt
import modules.mathematicalpendulum as mp

# Declare as floats to stop interference
time = [float()]
data = [float()]

# Import dataset
time, data = np.loadtxt(f'./data/data.txt', delimiter=",").T

# Set known data
l = 0.5

# Fitting data
IC = [0.25, 5, 3/180*np.pi] # best guess for the values
bounds = [[0,1],[0.1,10],[-np.pi,np.pi]] # bounds on the fitting function
estErr = 0.05 # estimated error in actual position

optimal, covariance = scipy.optimize.curve_fit(lambda t, A, b, p: mp.mathPendulum(t,A,b,l,p), time, data, p0=IC, bounds=bounds, sigma=estErr, absolute_sigma=True)

# Plot output

tSpace = np.linspace(0,np.max(time),10000)

plt.figure(figsize=[10,20])
plt.scatter(time,data, "k+")
plt.plot(tSpace, mp.mathPendulum(tSpace, optimal[0], optimal[1], l, optimal[2]))
plt.show()