import pymathtoolbox
import numpy as np

# Generate test data (5-dimensional points)
points = []
points.append(np.array([0.0, 2.0, 0.0, 3.0, 4.0]))
points.append(np.array([1.0, 0.0, 2.0, 4.0, 3.0]))
points.append(np.array([0.0, 1.0, 4.0, 2.0, 0.0]))
points.append(np.array([0.0, 4.0, 1.0, 0.0, 2.0]))
points.append(np.array([4.0, 3.0, 0.0, 1.0, 0.0]))
points.append(np.array([3.0, 4.0, 2.0, 0.0, 1.0]))
points.append(np.array([0.0, 2.0, 4.0, 1.0, 0.0]))
points.append(np.array([2.0, 0.0, 1.0, 4.0, 0.0]))
points.append(np.array([0.0, 1.0, 0.0, 3.0, 4.0]))
points.append(np.array([1.0, 0.0, 2.0, 0.0, 3.0]))

# Generate a distance matrix
D = np.zeros((10, 10));
for i in range(10):
    for j in range(10):
        d = np.linalg.norm(points[i] - points[j])
        D[i, j] = d
        D[j, i] = d

# Compute metric MDS (embedding into a 2-dimensional space)
X = pymathtoolbox.ComputeClassicalMds(D, 2);

# Show the result
print(X)
