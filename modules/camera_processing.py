import numpy as np

def true_position(raw_x, raw_y, pixel_size, resolution, focal_length, z_distance):
    true_x = []
    true_y = []

    centre = np.array(resolution) / 2 * pixel_size


    for coordinate in zip(raw_x, raw_y):
        x, y = np.array(coordinate) * np.array(pixel_size) - centre

        true_x.append(-((x * focal_length) / (z_distance - focal_length)))
        true_y.append(-((y * focal_length) / (z_distance - focal_length)))

    return true_x, true_y

def angle(V, P):
    return np.arctan( (V[0] - P[0]) / (V[1] - P[1]) )

