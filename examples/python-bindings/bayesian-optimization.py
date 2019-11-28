import pymathtoolbox
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns


# Define the objective function
def objective_func(x: np.ndarray) -> float:
    assert x.shape == (1, )
    return 1.0 - 1.5 * x[0] * math.sin(x[0] * 13.0)


# Initialize the random seed
pymathtoolbox.set_seed(random.randint(0, 65535))

# Define constants
NUM_ITERS = 15
CONFIDENT_REGION_ALPHA = 0.2
FIG_SIZE = (4, 6)
Y_RANGE = (-0.5, 2.5)
IMAGE_FORMAT = "png"
DPI = 150

# Define the bounding box
lower_bound = np.zeros(1)
upper_bound = np.ones(1)

# Instantiate the optimizer
optimizer = pymathtoolbox.BayesianOptimizer(objective_func, lower_bound,
                                            upper_bound)

# Set up the plot design
sns.set()
sns.set_context()
plt.rcParams['font.sans-serif'] = ["Linux Biolinum"]

for i in range(NUM_ITERS):
    # Proceed the optimization step
    x_new, y_new = optimizer.step()

    # Calculate sequences of relevant stats values
    x_samples = np.arange(0.0, 1.0, 0.001)

    vec_func = np.vectorize(lambda x: optimizer.predict_mean(np.array([x])))
    mean_values = vec_func(x_samples)

    vec_func = np.vectorize(lambda x: optimizer.predict_stdev(np.array([x])))
    stdev_values = vec_func(x_samples)

    lower_values = mean_values - stdev_values
    upper_values = mean_values + stdev_values

    vec_func = np.vectorize(lambda x: optimizer.calc_acquisition_value(np.array([x])))
    acquisition_values = vec_func(x_samples)

    # Prepare a figure object
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)

    # Begin to draw the top plot
    ax = fig.add_subplot(2, 1, 1)
    ax.set_title("Objective & Surrogate Functions")
    ax.set_ylim(Y_RANGE)

    # Plot the predicted confidence interval
    ax.fill_between(x_samples,
                    lower_values,
                    upper_values,
                    alpha=CONFIDENT_REGION_ALPHA)

    # Plot the predicted mean
    ax.plot(x_samples, mean_values)

    # Plot the target objective function
    vec_func = np.vectorize(lambda x: objective_func(np.array([x])))
    ax.plot(x_samples, vec_func(x_samples), linestyle="dashed")

    # Plot the new sampling
    ax.plot(x_new, y_new, marker='.')

    # Plot the current maximizer
    x_plus = optimizer.get_current_optimizer()
    ax.plot(x_plus, objective_func(x_plus), marker='o')

    # Begin to draw the bottom plot
    ax = fig.add_subplot(2, 1, 2)
    ax.yaxis.set_visible(False)
    ax.set_title("Acquisition Function")

    # Plot the acquisition function
    ax.plot(x_samples, acquisition_values)

    # Export the figure as an image file
    output_path = "./bayesian-optimization-" + "{:03}".format(i + 1) + "." + IMAGE_FORMAT

    fig.tight_layout()
    fig.savefig(output_path)
