import pymathtoolbox
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image
from scipy.spatial.distance import pdist, squareform

# Load an image
asset_dir_path = os.path.dirname(os.path.abspath(__file__)) + "/assets"
image = Image.open(asset_dir_path + "/autumn-leaves.jpg")
resized_image = image.resize((30, 20), Image.BILINEAR)

# Generate a color array
colors = np.asarray(resized_image)
colors = colors.reshape(colors.shape[0] * colors.shape[1], 3) / 255.0

# Instantiate a SOM object
som = pymathtoolbox.Som(data=colors.transpose(),
                        resolution=4,
                        latent_num_dims=1,
                        normalize_data=False)

for i in range(5):
    print(som.get_data_node_positions())
    som.step()
