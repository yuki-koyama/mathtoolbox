import pymathtoolbox
import numpy as np
from scipy.spatial.distance import pdist, squareform

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
D = squareform(pdist(points))

# Compute metric MDS (embedding into a 2-dimensional space)
X = pymathtoolbox.compute_classical_mds(D=D, target_dim=2)

# Show the result
print(X)
