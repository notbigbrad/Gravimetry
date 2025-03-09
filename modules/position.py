import numpy as np

def truePos(coord, pixelSize, resolution, f, z):
    
    truex = []
    truey = []
    
    # Find midpoint position
    centre = np.array(resolution)/2*pixelSize
    
    for i in np.array(coord).T:
        # Extract pixel coordinate
        p = np.array(i)

        # Convert pixel index to distance
        x, y = (p*pixelSize - centre)

        # Calculate true positions
        truex.append(x - (z*x)/(f))
        truey.append(y - (z*y)/(f))

    return truex, truey

def angle(V, P):
    return np.arctan( (V[0] - P[0]) / (V[1] - P[1]) )