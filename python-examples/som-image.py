import pymathtoolbox
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from PIL import Image

# Initialize the random seed
pymathtoolbox.set_seed(random.randint(0, 65535))

# Define constants for plot
FIG_SIZE = (8, 4)
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
resolution = 40
som = pymathtoolbox.Som(data=colors.transpose(),
                        resolution=resolution,
                        latent_num_dims=2,
                        init_var=0.50,
                        min_var=1e-03,
                        var_decreasing_speed=4.0,
                        normalize_data=True)

for i in range(30):

    previous_data_space_node_positions = som.get_data_space_node_positions()

    som.step()

    delta = np.linalg.norm(previous_data_space_node_positions -
                           som.get_data_space_node_positions())

    print("#iterations: {} (delta = {})".format(i + 1, delta))

    # Prepare a figure object
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

    # Draw the target image
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Target Image")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)

    # Draw the current self-organizing map
    ax = fig.add_subplot(1, 2, 2)
    title = "Self-Organizing Map of Pixel Colors"
    title = title + " [#iterations: {:02}]".format(i + 1)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    data_space_node_positions = som.get_data_space_node_positions()
    map = data_space_node_positions.transpose().reshape(
        resolution, resolution, 3)
    ax.imshow(map.clip(min=0.0, max=1.0), interpolation="bicubic")

    # Export plot
    fig.tight_layout()
    fig.savefig("./som-image-{:02}.{}".format(i, IMAGE_FORMAT))

    plt.close()
