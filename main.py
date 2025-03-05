import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from modules.position import *
from modules.model import *
from modules.error import *

# Perfrom pre-processing

resolution = [1280,720]
pixelSize = np.array([8.19,5.83]) / resolution
focalLength = 24 * 10 ** -3
zDist = [[0.725],[0.725],[0.709],[0.709],[0.826],[0.826]]
frameRate = 240
videoRate = 30

rawData = []

# files = os.listdir('./data/')

# for i in files[0:1]:
#     print(f'Adding data file: {i}')
#     rawData.append(np.loadtxt(f'./data/{i}', delimiter=",", encoding="utf-8-sig")[20:-60].T)

# Remove bad datapoints and leaving a whole number of waves
rawData.append(np.loadtxt(f'./data/1v1.txt', delimiter=",", encoding="utf-8-sig")[0:-210].T)
rawData.append(np.loadtxt(f'./data/1v2.txt', delimiter=",", encoding="utf-8-sig")[200:-10].T)
rawData.append(np.loadtxt(f'./data/15v1.txt', delimiter=",", encoding="utf-8-sig")[0:-220].T)
rawData.append(np.loadtxt(f'./data/15v2.txt', delimiter=",", encoding="utf-8-sig")[0:-90].T)
rawData.append(np.loadtxt(f'./data/2v1.txt', delimiter=",", encoding="utf-8-sig")[0:-310].T)
rawData.append(np.loadtxt(f'./data/2v2.txt', delimiter=",", encoding="utf-8-sig")[0:-250].T)

positionData = []

for i in range(len(rawData)):
    t = np.linspace(min(rawData[i][0]), max(rawData[i][0]), len(rawData[i][0])) / (frameRate/videoRate)
    p = truePos([rawData[i][1],rawData[i][2]], pixelSize=pixelSize, resolution=resolution, f=focalLength, z=zDist[i][0])
    positionData.append([t, p[0], p[1]])
    
zeroPoints = []

for i in positionData:
    zeroPoints.append(np.mean(i[1]))

# Now find angle subtended by the pendulum
    
# Find l
dBall = 0.03
# dBallStd = 0.0003
x = np.array([2.010])
d = np.array([0.485])/2
# xstd = 0.001
# dstd = 0.001
L = np.sqrt((x)**2 - (d)**2)
# Lstd = sqrt(L,x,d,xstd,dstd,0,"+")
l = np.mean(L)
# z = np.linspace(0, len(L) - 1, len(L))
# l, lcov = scipy.optimize.curve_fit(f, z, L, sigma=Lstd, absolute_sigma=True, maxfev=1*10**9)
l = l+dBall/2 # Add width of the ball
# lstd = add(1,1/2,np.sqrt(np.diag(lcov))[0],dBallStd,0)
# print(f'l: {l} +- {lstd} m')

pivots = []

for i in range(len(zeroPoints)):
    pivots.append([zeroPoints[i], np.mean(positionData[i][2] + l)])

print()

angularData = []

for i in range(len(positionData)):
    print(np.array(positionData[i][1:3]).shape)
    angularData.append([positionData[i][0], angle(np.array(positionData[i][1:3]), pivots[i])])

print(angularData[0])

for i in range(len(angularData)):
        plt.figure(figsize=[10,7])
        plt.axis("off")
        plt.subplot(211)
        plt.scatter(angularData[i][0], angularData[i][1], color="red", marker="+", linewidths=1, label="theta")
        plt.axhline(y=0, color="red", linestyle="-")
        plt.legend()
        plt.show()

# Fit model to data

# Propogate errors and find final value