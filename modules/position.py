import numpy as np
import scipy

def truePos(coord, pixelSize, resolution, f, z):
    
    truex = []
    truey = []
    
    # Find midpoint position
    centre = np.array(resolution)/2*pixelSize
    
    for i in np.array(coord).T:
        p = np.array(i)

        # Convert pixel index to distance
        x, y = (p*pixelSize - centre)

        # truex.append(-((x * f)/(z - f)))
        # truey.append(-((y * f)/(z - f)))
        truex.append(x - (z*x)/(f))
        truey.append(y - (z*y)/(f))

    return truex, truey

def angle(V, P):
    return np.arctan( (V[0] - P[0]) / (V[1] - P[1]) )