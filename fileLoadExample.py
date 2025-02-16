import numpy as np
import os

fileNames = os.listdir('./data')

time = [float()]
data = [float()]

for i in fileNames:
    raw = np.loadtxt(f'./data/{i}', delimiter=" ").T
    time.append(raw[0])
    data.append(raw[1])

print(time)
print(data)