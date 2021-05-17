import pymathtoolbox
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
FIG_SIZE = (8, 4)
Y_RANGE = (-0.5, 3.0)
IMAGE_FORMAT = "png"
DPI = 200

# Generate data points
large_x = np.array([[0.1, 0.4, 0.7, 0.9]])
small_y = np.array([1.0, 1.6, 0.8, 1.5])

# Set up the plot design
sns.set()
sns.set_context()
plt.rcParams['font.sans-serif'] = ["Linux Biolinum O", "Linux Biolinum"]

# Prepare a figure object
fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

# Define interpolation settings
conditions = [
    {
        "use_polynomial_term": False,
    },
    {
        "use_polynomial_term": True,
    },
]
regularization_weight = 1e-08

for index, condition in enumerate(conditions):
    use_polynomial_term = condition["use_polynomial_term"]

    rbf_kernels = [
        (pymathtoolbox.GaussianRbfKernel(10.0), "Gaussian"),
        (pymathtoolbox.ThinPlateSplineRbfKernel(), "thin plate spline"),
    ]

    # Begin to draw the plot
    ax = fig.add_subplot(1, 2, index + 1)
    ax.set_ylim(Y_RANGE)

    for i, rbf_kernel in enumerate(rbf_kernels):
        # Instantiate the interpolator
        interpolator = pymathtoolbox.RbfInterpolator(rbf_kernel[0],
                                                     use_polynomial_term)

        # Prepare interpolator
        interpolator.set_data(large_x, small_y)
        interpolator.calc_weights(True, regularization_weight)

        # Calculate sequences of interpolated values
        x_samples = np.arange(-0.8, 1.8, 0.001)
        vec_func = np.vectorize(
            lambda x: interpolator.calc_value(np.array([x])))
        values = vec_func(x_samples)

        # Plot the interpolated values
        ax.plot(x_samples,
                values,
                label="Interpolation ({})".format(rbf_kernel[1]),
                color=sns.color_palette()[1 + i])

    # Plot the observed sampling points
    ax.plot(np.transpose(large_x),
            small_y,
            marker="o",
            linewidth=0.0,
            markersize=4.0,
            label="Observed data",
            color=sns.color_palette()[0])

    # Show legends
    ax.legend(loc="upper left")

    # Set title
    title = "RBF Interpolation " + ("with" if use_polynomial_term else
                                    "without") + " Polynomial Term"
    ax.set_title(title)

# Tighten the layout
fig.tight_layout()

# Export the figure as an image file
output_path = "./rbf-interpolation-out." + IMAGE_FORMAT
fig.savefig(output_path)
