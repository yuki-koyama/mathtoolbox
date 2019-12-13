import pymathtoolbox
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns


# Define the objective function
def objective_func(x: np.ndarray) -> float:
    assert x.shape == (1, )
    return -1.5 * x[0] * math.sin(x[0] * 13.0)


# Define constants
NUM_SAMPLES = 50
FIG_SIZE = (8, 4)
Y_RANGE = (-1.2, 1.7)
IMAGE_FORMAT = "png"
DPI = 200
NOISE_INTENSITY = 0.1

# Generate data points
large_x = np.random.rand(1, NUM_SAMPLES)
small_y = np.ndarray(NUM_SAMPLES, )

for i in range(NUM_SAMPLES):
    noise = NOISE_INTENSITY * np.random.randn(1, )
    small_y[i] = objective_func(large_x[:, i]) + noise

# Set up the plot design
sns.set()
sns.set_context()
plt.rcParams['font.sans-serif'] = ["Linux Biolinum O", "Linux Biolinum"]

# Prepare a figure object
fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

# Define interpolation settings
conditions = [
    {
        "use_regularization": False,
    },
    {
        "use_regularization": True,
    },
]
regularization_weight = 1e-03
rbf_kernel = pymathtoolbox.ThinPlateSplineRbfKernel()

for index, condition in enumerate(conditions):
    use_regularization = condition["use_regularization"]

    # Instantiate the interpolator
    interpolator = pymathtoolbox.RbfInterpolator(rbf_kernel)

    # Prepare interpolator
    interpolator.set_data(large_x, small_y)
    interpolator.calc_weights(use_regularization, regularization_weight)

    # Calculate sequences of interpolated values
    x_samples = np.arange(0.0, 1.0, 0.001)
    vec_func = np.vectorize(lambda x: interpolator.calc_value(np.array([x])))
    values = vec_func(x_samples)

    # Begin to draw the plot
    ax = fig.add_subplot(1, 2, index + 1)
    ax.set_ylim(Y_RANGE)

    # Plot the observed sampling points
    ax.plot(np.transpose(large_x),
            small_y,
            marker="o",
            linewidth=0.0,
            markersize=4.0,
            label="Observed data",
            color=sns.color_palette()[0])

    # Plot the interpolated values
    ax.plot(x_samples,
            values,
            label="Interpolated values",
            color=sns.color_palette()[0])

    # Show legends
    ax.legend(loc="upper left")

    # Set title
    title = "RBF Interpolation " + ("with" if use_regularization else
                                    "without") + " Regularization"
    ax.set_title(title)

# Tighten the layout
fig.tight_layout()

# Export the figure as an image file
output_path = "./rbf-interpolation-out." + IMAGE_FORMAT
fig.savefig(output_path)
