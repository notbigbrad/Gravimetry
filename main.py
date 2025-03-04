import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from modules.error_and_evalutation import evaluation_with_error
from modules.position import *
from modules.model import *
from modules.error import *
import sympy as sp
from modules.errorProp import *

# Perfrom pre-processing

resolution = [720,1280]
pixelSize = np.array([7.3,9.8]) / resolution
focalLength = 24 * 10 ** -3
zDist = [
    0.725,0.725,
    0.709,0.709,
    0.826,0.826]
frameRate = 240
videoRate = 60

rawData = []

# files = os.listdir('./data/')

# for i in files[0:1]:
#     print(f'Adding data file: {i}')
#     rawData.append(np.loadtxt(f'./data/{i}', delimiter=",", encoding="utf-8-sig")[20:-60].T)

# Remove bad datapoints and leaving a whole number of waves
rawData.append(np.loadtxt(f'./data/2v1.txt', delimiter=",", encoding="utf-8-sig")[0:-310].T)
rawData.append(np.loadtxt(f'./data/2v2.txt', delimiter=",", encoding="utf-8-sig")[0:-250].T)
rawData.append(np.loadtxt(f'./data/15v1.txt', delimiter=",", encoding="utf-8-sig")[0:-220].T)
rawData.append(np.loadtxt(f'./data/15v2.txt', delimiter=",", encoding="utf-8-sig")[0:-90].T)
rawData.append(np.loadtxt(f'./data/1v1.txt', delimiter=",", encoding="utf-8-sig")[0:-210].T)
rawData.append(np.loadtxt(f'./data/1v2.txt', delimiter=",", encoding="utf-8-sig")[200:-10].T)

positionData = []

for i in range(len(rawData)):
    t = np.linspace(0, max(rawData[i][0]) - min(rawData[i][0]), len(rawData[i][0])) / (frameRate/videoRate)
    p = truePos([rawData[i][1],rawData[i][2]], pixelSize=pixelSize, resolution=resolution, f=focalLength, z=zDist[i])
    positionData.append([t, p[0], p[1]])
    
zeroPoints = []

for i in positionData:
    zeroPoints.append(np.mean(i[1]))

# Now find angle subtended by the pendulum
    
# Find l
dBall = 0.03
dBallStd = 0.0003
xRaw = [
    [1.999,1.999,1.998],[1.999,1.999,1.998],
    [1.499],[1.499],
    [1.016],[1.016]]
dRaw = [
    [0.376,0.374,0.375],[0.376,0.374,0.375],
    [0.375,0.376,0.377],[0.375,0.376,0.377],
    [0.380,0.378,0.379],[0.380,0.378,0.379]]

x = []
xStd = []

for i in xRaw:
    a, b = wghtMn(i)
    x.append(a)
    xStd.append(b)

d = []
dStd = []

for i in dRaw:
    a, b = wghtMn(i)
    d.append(a/2)
    dStd.append(b/2)

x_legs = np.array(x)
xStd = np.array(xStd)
d_horizontal = np.array(d)
dStd = np.array(dStd)


x_l, d = sp.symbols('x_l, d')
expr_l = sp.sqrt( (x_l) **2 -  (d) **2 )

print('-----------------------')

l_ = []
lStd = []

for i in range(len(x)):

    l, l_variance = evaluation_with_error(
        my_function=expr_l,
        legs=[(x_legs[i],xStd[i]),x_l],
        horizontal=[(d_horizontal[i],dStd[i]),d],
    )

    l_standard_deviation = np.sqrt(l_variance)


    l_.append(l)
    lStd.append(l_standard_deviation)
    print(l, l_standard_deviation)

print('------------------')

# l = np.sqrt((x)**2 - (d)**2)
# Lstd = sqrt(L,x,d,xstd,dstd,0,"-")

# SEM for l possibly

# l = l+dBall/2 # Add width of the ball
# lstd = add(1,1/2,np.sqrt(np.diag(lcov))[0],dBallStd,0)

pivots = []

for i in range(len(zeroPoints)):
    pivots.append([zeroPoints[i], np.mean(positionData[i][2] + l[i])])

angularData = []

for i in range(len(positionData)):
    print(np.array(positionData[i][1:3]).shape)
    angularData.append([positionData[i][0], angle(np.array(positionData[i][1:3]), pivots[i])])

# for i in range(len(angularData)):
#         plt.figure(figsize=[10,7])
#         plt.axis("off")
#         plt.subplot(211)
#         plt.scatter(angularData[i][0], angularData[i][1], color="red", marker="+", linewidths=1, label="theta")
#         plt.axhline(y=0, color="red", linestyle="-")
#         plt.legend()
#         plt.show()

# Fit model to data

for i in range(len(angularData)):

    time, theta = angularData[i]
        
    # Initial guesses
    I = [0.03, 0.1, np.sqrt(9.81/l[i]), np.pi/4] # A0, gamma, omega, phi

    # Bounds
    bounds = [[0.0001,0.0001,0.1,-np.pi],[0.04,1,10,np.pi]] # bounds on the fitting function

    # Fit model
    optimal, covariance = scipy.optimize.curve_fit(physicalPendulum, time, theta, p0=I, bounds=bounds, maxfev=1*10**9)
    # optimal, covariance = scipy.optimize.curve_fit(physicalODE, time, theta, p0=I, bounds=bounds, maxfev=1*10**9)

    # Find natural frequency
    # o^2 = o0^2 - gamma^2
    # o0 = np.sqrt(optimal[2]**2 + optimal[1]**2)
    o0 = optimal[2]
    # o0Std = sqrt(o0,optimal[2],optimal[1],np.sqrt(covariance[2,2]),np.sqrt(covariance[1,1]),0, "+") # cov should be np.sqrt(covariance[1,2]) instead 0

    # Calculate g
    # o0^2 = mgro+ / I0
    # g = o0^2*I0 / mro+
    g = float()
    gStd = float()
    g = (o0**2)*l[i]

    # gStd = mul(g,o0**2,l,squared(o0,o0Std),lstd,0) # Assuming omega0 and l are not covariant

    print(f'g: {g} +- {gStd} ms^-2')

    # Calculate residuals
    r = theta - physicalPendulum(np.linspace(np.min(time),np.max(time),len(time)), optimal[0], optimal[1], optimal[2], optimal[3])

    # Plot output
    tSpace = np.linspace(np.min(time),np.max(time),10000)
    plt.figure(figsize=[10,7])
    plt.suptitle("plt")
    plt.axis("off")
    plt.subplot(211)
    plt.title("Data Plot with Fitted Model")
    plt.fill_between(time, theta-0, theta+0, color="lightgray")
    plt.scatter(time, theta, color="red", marker="+", linewidths=1, label="Data")
    plt.plot(tSpace, physicalPendulum(tSpace, optimal[0], optimal[1], optimal[2], optimal[3]), label="Mathematical Pendulum Model")
    plt.plot(tSpace, max(theta)*sin(tSpace, np.sqrt(9.81/l[i]), 0), "g--", label="Theoretical")
    plt.legend()
    plt.subplot(212)
    plt.title("Residual")
    plt.plot(time, theta*0.05, color="lightgray")
    plt.plot(time, r, color="black")
    plt.show()

# Propogate errors and find final value