import numpy as np
import scipy

def truePos(coord, pixelSize, resolution, f, z):
    # Find midpoint position
    centre = np.array(resolution)/2*pixelSize

    # Convert pixel index to distance
    x, y = (coord*pixelSize - centre)

    truex = -((x * f)/(z - f))
    truey = -((y * f)/(z - f))

    return [truex, truey]

res = [1920,1080]
pixel = np.array([8.19,5.83])/res
focalLength = 0.024
zDist = 0.1

print(truePos([200,200], pixel, res, focalLength, zDist))
