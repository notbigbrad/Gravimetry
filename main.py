import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy as sp
import modules.monteCarlo as monteCarlo
from modules.error_and_evalutation import evaluation_with_error
from modules.position import *
from modules.model import *
from modules.error import *
from datetime import datetime as dt
import re
plt.show = lambda : 0 # dissable plot output

# ============================================================================= Logging

class DualStream:
    def __init__(self):
        fName = re.sub(r'[<>:"/\\|?*]', '-', dt.now().strftime('%Y-%m-%d %H;%M;%S.%f'))
        print(f'Printing log to: {fName}')
        self.file = open(f'./logs/{fName}.txt', 'w')
        self.console = sys.stdout

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

sys.stdout = DualStream()

# ============================================================================= Perfrom pre-processing

resolution = [720,1280]
pixelSize = np.array([7.3*10**-3,9.8*10**-3]) / resolution
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

# ============================================================================= Now find angle subtended by the pendulum
    
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
rawLengthStd = 0.001

x = []
xStd = []

for i in xRaw:
    a, b = weightedMean(i, rawLengthStd)
    x.append(a)
    xStd.append(b)

d = []
dStd = []

for i in dRaw:
    a, b = weightedMean(i, rawLengthStd)
    d.append(a/2)
    dStd.append(b/2)
    
x = np.array(x)
xStd = np.array(xStd)
d = np.array(d)
dStd = np.array(dStd)

xSym, dSym, dBallSym = sp.symbols('xSym, dSym, dBallSym')
expr_length = sp.sqrt( (xSym) **2 -  (dSym) **2 ) + dBallSym/2

l = []
lStd = []

for i in range(len(x)):

    l_curr, l_std = evaluation_with_error(
        my_function=expr_length,
        legs=[(x[i],xStd[i]),xSym],
        horizontal=[(d[i],dStd[i]),dSym],
        ball=[(dBall,dBallStd),dBallSym],
    )

    l.append(l_curr)
    lStd.append(l_std)
    print(f'l: {l_curr:.5f} +- {l_std:.5f} m')

pivots = []

for i in range(len(zeroPoints)):
    pivots.append([zeroPoints[i], np.mean(positionData[i][2] + l[i])])

angularData = []

for i in range(len(positionData)):
    angularData.append([positionData[i][0], angle(np.array(positionData[i][1:3]), pivots[i])])

# ============================================================================= Calculate constants with error

m = 0.109
mStd = 0.002

lSym, dBallSym = sp.symbols('dSym, dBallSym')
expr_r = lSym

r0 = []
r0Std = []

for i in range(len(l)):

    r_curr, r_std = evaluation_with_error(
        my_function=expr_r,
        ball=[(dBall,dBallStd),dBallSym],
        length=[(l[i],lStd[i]),lSym],
    )

    r0.append(r_curr)
    r0Std.append(r_std)
    print(f'r0: {r_curr:.5f} +- {r_std:.5f} kgm^2')

mSym, lSym, dBallSym = sp.symbols('xSym, dSym, dBallSym')
expr_moment = (2/5)*mSym*(dBallSym/2)**2 + mSym*(lSym)**2

I0 = []
I0Std = []

for i in range(len(l)):

    i_curr, i_std = evaluation_with_error(
        my_function=expr_moment,
        ball=[(dBall,dBallStd),dBallSym],
        mass=[(m,mStd),mSym],
        length=[(l[i],lStd[i]),lSym],
    )

    I0.append(i_curr)
    I0Std.append(i_std)
    print(f'I: {i_curr:.5f} +- {i_std:.5f} kgm^2')

# ============================================================================= Fit model to data

g = []
gStd = []

gODE = []
gStdODE = []

gMonte = []
gStdMonte = []

for i in range(len(angularData)):

    time, theta = angularData[i]
        
    # Initial guesses
    I1 = [0.03, 0.1, np.sqrt(9.81/l[i]), np.pi/4] # A0, gamma, omega, phi MATH
    I2 = [np.pi/4, 0, 9.816, 0.0001] # thet0, om0, g, b ODE

    # Bounds
    bounds1 = [[0.0001,0.0001,0.1,-np.pi],[0.04,1,10,np.pi]] # bounds on the fitting function MATH
    bounds2 = [[-np.pi,0,7,0.00001],[np.pi,5,12,0.1]] # bounds on the fitting function ODE
    
    # Fit model
    optimal1, covariance1 = scipy.optimize.curve_fit(physicalPendulum, time, theta, p0=I1, bounds=bounds1, maxfev=1*10**9)
    optimal2, covariance2 = scipy.optimize.curve_fit(lambda t, thet0, om0, g, b : physicalODE(t, thet0, om0, g, b, m=m, r=r0[i], I=I0[i]), time, theta, p0=I2, bounds=bounds2, maxfev=1*10**9)
    results, errors, final = monteCarlo.prop(physicalODE, time, theta, bounds2, I2, (m, r0[i], I0[i]), (mStd, r0Std[i], I0Std[i]))

    o = optimal1[2]
    oStd = np.sqrt(covariance1[2,2])
    gamma = optimal1[1]
    gammaStd = np.sqrt(covariance1[1,1])

    # Calculate g
    oSym, gammaSym, lSym = sp.symbols('o0Sym, gammaSym, lSym')
    expr_g = (oSym**2 + gammaSym**2)*lSym

    g_curr, g_std = evaluation_with_error(
        my_function=expr_g,
        omega=[(o,oStd),oSym],
        gamma=[(gamma,gammaStd),gammaSym],
        length=[(l[i],lStd[i]),lSym],
    )
    
    g.append(g_curr)
    gStd.append(g_std)
    
    gODE.append(optimal2[2])
    gStdODE.append(np.sqrt(covariance2[2,2]))
    
    gMonte.append(final[0])
    gStdMonte.append(final[1])

    print(f'g: {g[i]:.5f} +- {gStd[i]:.5f} ms^-2 (math)')
    print(f'g: {gODE[i]:.5f} +- {gStdODE[i]:.5f} ms^-2 (phys)')
    print(f'g: {gMonte[i]:.5f} +- {gStdMonte[i]:.5f} ms^-2 (monte)')

    # Calculate residuals
    r1 = theta - physicalPendulum(np.linspace(np.min(time),np.max(time),len(time)), optimal1[0], optimal1[1], optimal1[2], optimal1[3])
    r2 = theta - physicalODE(np.linspace(np.min(time),np.max(time),len(time)), optimal2[0], optimal2[1], optimal2[2], optimal2[3], m, r0[i], I0[i])

    # Plot output
    tSpace = np.linspace(np.min(time),np.max(time),10000)
    plt.figure(figsize=[14,7])
    plt.suptitle("Data Plot with Fitted Model")
    plt.axis("off")
    plt.subplot(321)
    plt.title("Mathematical Pendulum Model")
    plt.fill_between(time, theta-0, theta+0, color="lightgray")
    plt.scatter(time, theta, color="red", marker="+", linewidths=1, label="Data")
    plt.plot(tSpace, physicalPendulum(tSpace, optimal1[0], optimal1[1], optimal1[2], optimal1[3]), label="Mathematical Pendulum Model")
    plt.subplot(322)
    plt.title("Physical Pendulum Model")
    plt.fill_between(time, theta-0, theta+0, color="lightgray")
    plt.scatter(time, theta, color="red", marker="+", linewidths=1, label="Data")
    plt.plot(tSpace, physicalODE(tSpace, optimal2[0], optimal2[1], optimal2[2], optimal2[3], m, r0[i], I0[i]), label="Physical Pendulum Model")
    plt.subplot(323)
    plt.title("Residual (Math)")
    plt.plot(time, theta*0.05, color="lightgray")
    plt.plot(time, r1, color="black")
    plt.subplot(324)
    plt.title("Residual (Phys)")
    plt.plot(time, theta*0.05, color="lightgray")
    plt.plot(time, r2, color="black")
    plt.subplot(325)
    plt.title("Residual (Math-Phys)")
    plt.plot(r1-r2, color="black")
    plt.subplot(326)
    plt.title("Monte Carlo")
    plt.errorbar(np.linspace(np.min(results),np.max(results),len(results)), results, yerr=errors, color="red")
    plt.plot(np.linspace(np.min(results),np.max(results),len(results)), f(np.linspace(np.min(results),np.max(results),len(results)), final[0]))
    plt.show()

# ============================================================================= Propogate errors and find final value

q = np.linspace(0, len(g) - 1, len(g))
optimal, covariance = scipy.optimize.curve_fit(f, q, g, p0=[9.81616], sigma=gStd, maxfev=1*10**9)
GStd = np.sqrt(covariance[0,0])
G = optimal[0]
optimal, covariance = scipy.optimize.curve_fit(f, q, gODE, p0=[9.81616], sigma=gStdODE, maxfev=1*10**9)
GStdODE = np.sqrt(covariance[0,0])
GODE = optimal[0]
optimal, covariance = scipy.optimize.curve_fit(f, q, gMonte, p0=[9.81616], sigma=gStdMonte, maxfev=1*10**9)
GStdMonte = np.sqrt(covariance[0,0])
GMonte = optimal[0]

print(f'Final value of g using all experiments: {G:.5f} +- {GStd:.5f} ms^-2 (math)')
print(f'Final value of g using all experiments: {GODE:.5f} +- {GStdODE:.5f} ms^-2 (phys)')
print(f'Final value of g using all experiments: {GMonte:.5f} +- {GStdMonte:.5f} ms^-2 (monte)')