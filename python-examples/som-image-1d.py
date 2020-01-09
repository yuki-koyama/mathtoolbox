import pymathtoolbox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import random
import seaborn as sns
from PIL import Image

# Initialize the random seed
pymathtoolbox.set_seed(random.randint(0, 65535))

# Define constants for plot
FIG_SIZE = (8, 2)
IMAGE_FORMAT = "png"
DPI = 200

# Set plot style
sns.set()
sns.set_context()
plt.rcParams['font.sans-serif'] = ["Linux Biolinum O", "Linux Biolinum"]

# Load an image
asset_dir_path = os.path.dirname(os.path.abspath(__file__)) + "/assets"
image = Image.open(asset_dir_path + "/autumn-leaves.jpg")
resized_image = image.resize((90, 60), Image.BILINEAR)

# Generate a color array
colors = np.asarray(resized_image)
colors = colors.reshape(colors.shape[0] * colors.shape[1], 3) / 255.0

# Instantiate a SOM object
resolution = 250
som = pymathtoolbox.Som(data=colors.transpose(),
                        resolution=resolution,
                        latent_num_dims=1,
                        init_var=0.50,
                        min_var=5e-04,
                        var_decreasing_speed=10.0,
                        normalize_data=True)

for i in range(250):

    previous_data_space_node_positions = som.get_data_space_node_positions()

    som.step()

    delta = np.linalg.norm(previous_data_space_node_positions -
                           som.get_data_space_node_positions())

    print("#iterations: {} (delta = {})".format(i + 1, delta))

# Prepare a figure object
fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

# Define a grid
gs = GridSpec(nrows=1, ncols=2, width_ratios=[9, 22])

# Draw the target image
ax = fig.add_subplot(gs[0])
ax.set_title("Target Image")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(image)

# Draw the current self-organizing map
ax = fig.add_subplot(gs[1])
title = "One-Dimensional Self-Organizing Map of Pixel Colors"
ax.set_title(title)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

data_space_node_positions = som.get_data_space_node_positions()
map = data_space_node_positions.transpose().reshape(1, resolution, 3)
ax.imshow(map.clip(min=0.0, max=1.0), aspect="auto", interpolation="bicubic")

# Export plot
fig.tight_layout()
fig.savefig("./som-image-1d.{}".format(IMAGE_FORMAT))

plt.close()
